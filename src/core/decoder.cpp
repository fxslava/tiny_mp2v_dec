// Copyright � 2021 Vladislav Ovchinnikov. All rights reserved.
#include <string.h>
#include "mb_decoder.h"
#include "decoder.h"
#include "mp2v_hdr.h"
#include "mp2v_vlc.h"
#include "misc.hpp"
#include "start_codes_search.hpp"

#define CHECK(p) { if (!(p)) return false; }

struct spatial_temporal_weights_classes_t {
    uint8_t spatial_temporal_weight_fract[2]; // 0 - 0.0, 1 - 0.5, 2 - 1.0
    uint8_t spatial_temporal_weight_class;
    uint8_t spatial_temporal_integer_weight;
};

//[spatial_temporal_weight_code_table_index][spatial_temporal_weight_code]
spatial_temporal_weights_classes_t local_spatial_temporal_weights_classes_tbl[4][4] = {
    {{{1, 1}, 1, 0}, {{1, 1}, 1, 0}, {{1, 1}, 1, 0}, {{1, 1}, 1, 0} },
    {{{0, 2}, 3, 1}, {{0, 1}, 1, 0}, {{1, 2}, 3, 0}, {{1, 1}, 1, 0} },
    {{{2, 0}, 2, 1}, {{1, 0}, 1, 0}, {{2, 1}, 2, 0}, {{1, 1}, 1, 0} },
    {{{2, 0}, 2, 1}, {{2, 1}, 2, 0}, {{1, 2}, 3, 0}, {{1, 1}, 1, 0} }
};

frame_c::frame_c(int width, int height, int chroma_format) {
    m_stride[0] = (width + CACHE_LINE - 1) & ~(CACHE_LINE - 1);
    m_width [0] = width;
    m_height[0] = height;

    switch (chroma_format) {
    case chroma_format_420:
        m_stride[1] = ((m_stride[0] >> 1) + CACHE_LINE - 1) & ~(CACHE_LINE - 1);
        m_width [1] = m_width[0] >> 1;
        m_height[1] = m_height[0] >> 1;
        break;
    case chroma_format_422:
        m_stride[1] = ((m_stride[0] >> 1) + CACHE_LINE - 1) & ~(CACHE_LINE - 1);
        m_width [1] = m_width[0] >> 1;
        m_height[1] = m_height[0];
        break;
    case chroma_format_444:
        m_stride[1] = m_stride[0];
        m_width [1] = m_width [0];
        m_height[1] = m_height[0];
        break;
    }
    m_stride[2] = m_stride[1];
    m_width [2] = m_width [1];
    m_height[2] = m_height[1];

#if defined(_MSC_VER)
    for (int i = 0; i < 3; i++)
        m_planes[i] = (uint8_t*)_aligned_malloc(m_height[i] * m_stride[i], 32);
#else
    for (int i = 0; i < 3; i++)
        m_planes[i] = (uint8_t*)aligned_alloc(32, m_height[i] * m_stride[i]);
#endif
}

frame_c::~frame_c() {
#if defined(_MSC_VER)
    for (int i = 0; i < 3; i++)
        if (m_planes[i]) _aligned_free(m_planes[i]);
#else
    for (int i = 0; i < 3; i++)
        if (m_planes[i]) free(m_planes[i]);
#endif
}

static void make_macroblock_yuv_ptrs(uint8_t* (&yuv)[3], frame_c* frame, int mb_row, int stride, int chroma_stride, int chroma_format) {
    yuv[0] = frame->get_planes(0) + mb_row * 16 * stride;// +mb_col * 16;
    switch (chroma_format) {
    case chroma_format_420:
        yuv[1] = frame->get_planes(1) + mb_row * 8 * chroma_stride;// +mb_col * 8;
        yuv[2] = frame->get_planes(2) + mb_row * 8 * chroma_stride;// +mb_col * 8;
        break;
    case chroma_format_422:
        yuv[1] = frame->get_planes(1) + mb_row * 16 * chroma_stride;// +mb_col * 8;
        yuv[2] = frame->get_planes(2) + mb_row * 16 * chroma_stride;// +mb_col * 8;
        break;
    case chroma_format_444:
        yuv[1] = frame->get_planes(1) + mb_row * 16 * chroma_stride;// +mb_col * 16;
        yuv[2] = frame->get_planes(2) + mb_row * 16 * chroma_stride;// +mb_col * 16;
        break;
    }
}

bool mp2v_picture_c::decode_slice(bitstream_reader_c bs) {
    auto& pcext = m_picture_coding_extension;
    auto& sh = m_dec->m_sequence_header;
    auto& sext = m_dec->m_sequence_extension;
    slice_t slice = { 0 };

    // decode slice header
    parse_slice_header(&bs, slice, sh, m_dec->m_sequence_scalable_extension);

    // calculate row position of the slice
    int mb_row = 0;
    int slice_vertical_position = slice.slice_start_code & 0xff;
    if (sh.vertical_size_value > 2800)
        mb_row = (slice.slice_vertical_position_extension << 7) + slice_vertical_position - 1;
    else
        mb_row = slice_vertical_position - 1;

    // fill cache
    auto refs = m_dec->ref_frames;
    macroblock_context_cache_t cache;
    memcpy(cache.W, quantiser_matrices, sizeof(cache.W));
    memcpy(cache.f_code, pcext.f_code, sizeof(cache.f_code));
    memset(cache.PMVs, 0, sizeof(cache.PMVs));
    for (auto& pred : cache.dct_dc_pred) pred = 1 << (pcext.intra_dc_precision + 7);
    cache.spatial_temporal_weight_code_table_index = 0;
    cache.luma_stride      = m_frame->m_stride[0];
    cache.chroma_stride    = m_frame->m_stride[1];
    cache.intra_dc_prec    = m_picture_coding_extension.intra_dc_precision;
    cache.intra_vlc_format = pcext.intra_vlc_format;
    cache.previous_mb_type = 0;
                 make_macroblock_yuv_ptrs(cache.yuv_planes[REF_TYPE_SRC], m_frame, mb_row, cache.luma_stride, cache.chroma_stride, sext.chroma_format);
    if (refs[0]) make_macroblock_yuv_ptrs(cache.yuv_planes[REF_TYPE_L0 ], refs[0], mb_row, cache.luma_stride, cache.chroma_stride, sext.chroma_format);
    if (refs[1]) make_macroblock_yuv_ptrs(cache.yuv_planes[REF_TYPE_L1 ], refs[1], mb_row, cache.luma_stride, cache.chroma_stride, sext.chroma_format);
    if (m_picture_coding_extension.q_scale_type) {
        if (slice.quantiser_scale_code < 9)       cache.quantiser_scale =  slice.quantiser_scale_code;
        else if (slice.quantiser_scale_code < 17) cache.quantiser_scale = (slice.quantiser_scale_code - 4) << 1;
        else if (slice.quantiser_scale_code < 25) cache.quantiser_scale = (slice.quantiser_scale_code - 10) << 2;
        else                                      cache.quantiser_scale = (slice.quantiser_scale_code - 17) << 3; }
    else                                          cache.quantiser_scale =  slice.quantiser_scale_code << 1;

    // decode macroblocks
    do {
        m_parse_macroblock_func(&bs, cache);
    } while (bs.get_next_bits(23) != 0);
    return true;
}

void mp2v_picture_c::init() {
    auto& sext = m_dec->m_sequence_extension;
    auto& pcext = m_picture_coding_extension;
    auto& ph = m_picture_header;
    m_parse_macroblock_func = select_parse_macroblock_func(
        ph.picture_coding_type,
        pcext.picture_structure,
        pcext.frame_pred_frame_dct,
        pcext.concealment_motion_vectors,
        sext.chroma_format,
        pcext.q_scale_type,
        pcext.alternate_scan);

    uint8_t tmp[4][64];
    if (m_quant_matrix_extension) {
        for (int i = 0; i < 64; i++) {
            int j = g_scan[0][i];
            if (m_quant_matrix_extension->load_intra_quantiser_matrix)            tmp[0][i] = m_quant_matrix_extension->intra_quantiser_matrix[j];
            if (m_quant_matrix_extension->load_non_intra_quantiser_matrix)        tmp[1][i] = m_quant_matrix_extension->non_intra_quantiser_matrix[j];
            if (m_quant_matrix_extension->load_chroma_intra_quantiser_matrix)     tmp[2][i] = m_quant_matrix_extension->chroma_intra_quantiser_matrix[j];
            if (m_quant_matrix_extension->load_chroma_non_intra_quantiser_matrix) tmp[3][i] = m_quant_matrix_extension->chroma_non_intra_quantiser_matrix[j];
        }
        for (int i = 0; i < 64; i++) {
            int j = g_shuffle[pcext.alternate_scan][i];
            if (m_quant_matrix_extension->load_intra_quantiser_matrix)            quantiser_matrices[0][i] = tmp[0][j];
            if (m_quant_matrix_extension->load_non_intra_quantiser_matrix)        quantiser_matrices[1][i] = tmp[1][j];
            if (m_quant_matrix_extension->load_chroma_intra_quantiser_matrix)     quantiser_matrices[2][i] = tmp[2][j];
            if (m_quant_matrix_extension->load_chroma_non_intra_quantiser_matrix) quantiser_matrices[3][i] = tmp[3][j];
        }
    }
}

bool mp2v_decoder_c::decode_user_data() {
    while (m_bs.get_next_bits(vlc_start_code.len) != vlc_start_code.value) {
        uint8_t data = m_bs.read_next_bits(8);
        user_data.push_back(data);
    }
    return true;
}

bool mp2v_decoder_c::decode_extension_data(mp2v_picture_c* pic) {
    m_bs.skip_bits(32);
    uint8_t ext_id = m_bs.get_next_bits(4);
    switch (ext_id)
    {
    case sequence_extension_id:
        parse_sequence_extension(&m_bs, m_sequence_extension); // <--
        break;
    case sequence_display_extension_id:
        parse_sequence_display_extension(&m_bs, *(m_sequence_display_extension = new sequence_display_extension_t));
        break;
    case sequence_scalable_extension_id:
        parse_sequence_scalable_extension(&m_bs, *(m_sequence_scalable_extension = new sequence_scalable_extension_t));
        break;
    case quant_matrix_extension_id:
        parse_quant_matrix_extension(&m_bs, *(pic->m_quant_matrix_extension = new quant_matrix_extension_t));
        break;
    case copiright_extension_id:
        parse_copyright_extension(&m_bs, *(pic->m_copyright_extension = new copyright_extension_t));
        break;
    case picture_coding_extension_id:
        parse_picture_coding_extension(&m_bs, pic->m_picture_coding_extension);
        break;
    case picture_display_extension_id:
        parse_picture_display_extension(&m_bs, *(pic->m_picture_display_extension = new picture_display_extension_t), m_sequence_extension, pic->m_picture_coding_extension);
        break;
    case picture_spatial_scalable_extension_id:
        parse_picture_spatial_scalable_extension(&m_bs, *(pic->m_picture_spatial_scalable_extension = new picture_spatial_scalable_extension_t));
        break;
    case picture_temporal_scalable_extension_id:
        parse_picture_temporal_scalable_extension(&m_bs, *(pic->m_picture_temporal_scalable_extension = new picture_temporal_scalable_extension_t));
        break;
    case picture_camera_parameters_extension_id:
        //parse_camera_parameters_extension();
        break;
    default:
        // Unsupported extension id (skip)
        break;
    }
    return true;
}

void mp2v_decoder_c::flush_mini_gop() {
    if (ref_frames[1])
        push_frame(ref_frames[1]);
}

mp2v_picture_c* mp2v_decoder_c::new_pic() {
    mp2v_picture_c* res = nullptr;
    frame_c* frame = nullptr;
    m_frames_pool.pop(frame);
    res = m_pictures_pool.back();
    res->attach(frame);
    m_pictures_pool.pop_back();
    return res;
}

void mp2v_decoder_c::out_pic(mp2v_picture_c* cur_pic) {
    auto* frame = cur_pic->get_frame();
    if (cur_pic->m_picture_header.picture_coding_type == picture_coding_type_bidir || !reordering)
        push_frame(frame);
    else {
        if (ref_frames[0])
            push_frame(ref_frames[0]);
    }
    cur_pic->attach(nullptr);
    m_pictures_pool.push_back(cur_pic);
}

bool mp2v_decoder_c::decode(uint8_t* buffer, int len) {

    m_bs.set_bitstream_buffer(buffer);
    bool new_picture = false, sequence_end = false;
    mp2v_picture_c* cur_pic = nullptr;

    scan_start_codes(buffer, buffer + len, [&](uint8_t* ptr) {
        BITSTREAM((&m_bs));
        bit_idx = 32;
        bit_ptr = (uint32_t*)(ptr + 4);
        bit_buf = (uint64_t)bswap_32(*((uint32_t*)ptr));
        uint8_t start_code = *(ptr + 3);
        switch (start_code) {
        case sequence_header_code: parse_sequence_header(&m_bs, m_sequence_header); break;
        case extension_start_code: decode_extension_data(cur_pic);                  break;
        case group_start_code:     parse_group_of_pictures_header(&m_bs, *(m_group_of_pictures_header = new group_of_pictures_header_t)); break;
        case picture_start_code: {
                new_picture = true;
                if (cur_pic) out_pic(cur_pic);
                cur_pic = new_pic();
                parse_picture_header(&m_bs, cur_pic->m_picture_header);
                if (cur_pic->m_picture_header.picture_coding_type == picture_coding_type_pred || cur_pic->m_picture_header.picture_coding_type == picture_coding_type_intra) {
                    ref_frames[0] = ref_frames[1];
                    ref_frames[1] = cur_pic->get_frame();
                }
            }
            break;
        case user_data_start_code: decode_user_data(); break;
        case sequence_error_code:
        case sequence_end_code:
            flush_mini_gop();
            push_frame(nullptr);
            sequence_end = true;
            break;
        default:
            if ((start_code >= slice_start_code_min) && (start_code <= slice_start_code_max)) {
                if (new_picture) cur_pic->init();
                cur_pic->decode_slice(m_bs);
                new_picture = false;
            }
        }
    });
    if (!sequence_end) {
        flush_mini_gop();
        push_frame(nullptr);
    }
    return true;
}

#ifdef MP2V_MT
void mp2v_decoder_c::threadpool_task_scheduler(mp2v_decoder_c* dec) {
    mp2v_slice_task_c* slice_task = nullptr;
    while (dec->task_queue->get_task((slice_task_c*&)slice_task) == TASK_QUEUE_SUCCESS) {
        
        slice_task->done();
    }
}
#endif

bool mp2v_decoder_c::decoder_init(decoder_config_t* config) {
    int pool_size = config->frames_pool_size;
    int num_pics = config->pictures_pool_size;
    int width = config->width;
    int height = config->height;
    int chroma_format = config->chroma_format;
    reordering = config->reordering;

    for (int i = 0; i < pool_size; i++)
        m_frames_pool.push(new frame_c(width, height, chroma_format));

#ifdef MP2V_MT
    task_queue = new task_queue_c(num_pics, [this]() -> picture_task_c* { return new mp2v_picture_c(this, nullptr); });
    num_threads = config->num_threads;
    for (int i = 0; i < num_threads; i++)
        thread_pool[i] = new std::thread(threadpool_task_scheduler, this);
#else
    for (int i = 0; i < num_pics; i++)
        m_pictures_pool.push_back(new mp2v_picture_c(this, nullptr));
#endif

    return true;
}

mp2v_decoder_c::~mp2v_decoder_c() {
    for (auto* pic : m_pictures_pool)
        delete pic;
#ifdef MP2V_MT
    for (auto*& thread : thread_pool)
        if (thread) {
            delete thread;
            thread = nullptr;
        }
#endif
}