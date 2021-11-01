// Copyright � 2021 Vladislav Ovchinnikov. All rights reserved.
#include <string.h>
#include "mb_decoder.h"
#include "decoder.h"
#include "mp2v_hdr.h"
#include "mp2v_vlc.h"
#include "misc.hpp"
#include "start_codes_search.hpp"

uint8_t default_intra_quantiser_matrix[64] = {
    8,  16, 19, 22, 26, 27, 29, 34,
    16, 16, 22, 24, 27, 29, 34, 37,
    19, 22, 26, 27, 29, 34, 34, 38,
    22, 22, 26, 27, 29, 34, 37, 40,
    22, 26, 27, 29, 32, 35, 40, 48,
    26, 27, 29, 32, 35, 40, 48, 58,
    26, 27, 29, 34, 38, 46, 56, 69,
    27, 29, 35, 38, 46, 56, 69, 83 };

uint8_t default_non_intra_quantiser_matrix[64] = {
    16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16 };

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
    mp2v_picture_c* refs[2] = {(mp2v_picture_c*)dependencies[0], (mp2v_picture_c*)dependencies[1]};
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
    if (refs[0]) make_macroblock_yuv_ptrs(cache.yuv_planes[REF_TYPE_L0 ], refs[0]->get_frame(), mb_row, cache.luma_stride, cache.chroma_stride, sext.chroma_format);
    if (refs[1]) make_macroblock_yuv_ptrs(cache.yuv_planes[REF_TYPE_L1 ], refs[1]->get_frame(), mb_row, cache.luma_stride, cache.chroma_stride, sext.chroma_format);
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

    bool def_mat = m_quant_matrix_extension != nullptr;
    uint8_t tmp[4][64];
    for (int i = 0; i < 64; i++) {
        int j = g_scan[0][i];
        tmp[0][i] = default_intra_quantiser_matrix[j];
        tmp[1][i] = default_non_intra_quantiser_matrix[j];
        tmp[2][i] = default_intra_quantiser_matrix[j];
        tmp[3][i] = default_non_intra_quantiser_matrix[j];
    }
    if (def_mat) {
        for (int i = 0; i < 64; i++) {
            int j = g_scan[0][i];
            if (m_quant_matrix_extension->load_intra_quantiser_matrix)            tmp[0][i] = m_quant_matrix_extension->intra_quantiser_matrix[j];
            if (m_quant_matrix_extension->load_non_intra_quantiser_matrix)        tmp[1][i] = m_quant_matrix_extension->non_intra_quantiser_matrix[j];
            if (m_quant_matrix_extension->load_chroma_intra_quantiser_matrix)     tmp[2][i] = m_quant_matrix_extension->chroma_intra_quantiser_matrix[j];
            if (m_quant_matrix_extension->load_chroma_non_intra_quantiser_matrix) tmp[3][i] = m_quant_matrix_extension->chroma_non_intra_quantiser_matrix[j];
        }
    }
    for (int i = 0; i < 64; i++) {
        int j = g_shuffle[pcext.alternate_scan][i];
        if (m_quant_matrix_extension->load_intra_quantiser_matrix)            quantiser_matrices[0][i] = tmp[0][j];
        if (m_quant_matrix_extension->load_non_intra_quantiser_matrix)        quantiser_matrices[1][i] = tmp[1][j];
        if (m_quant_matrix_extension->load_chroma_intra_quantiser_matrix)     quantiser_matrices[2][i] = tmp[2][j];
        if (m_quant_matrix_extension->load_chroma_non_intra_quantiser_matrix) quantiser_matrices[3][i] = tmp[3][j];
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

void mp2v_decoder_c::flush() {
#ifdef MP2V_MT
    if (cur_pic) {
        task_queue->add_task(cur_pic, cur_pic->m_picture_header.picture_coding_type == picture_coding_type_bidir);
        cur_pic = nullptr;
    }
    task_queue->kill();
#else
    if (ref_frames[1])
        m_done_pics.push(ref_frames[1]);
    m_done_pics.push(nullptr);
#endif
}

mp2v_picture_c* mp2v_decoder_c::new_pic() {
    mp2v_picture_c* res = nullptr;
#ifdef MP2V_MT
    res = (mp2v_picture_c*)task_queue->create_task();
#else
    m_free_pics.pop(res);
    res->reset();
#endif
    return res;
}

void mp2v_decoder_c::out_pic(mp2v_picture_c* cur_pic) {
#ifdef MP2V_MT
    task_queue->add_task(cur_pic, cur_pic->m_picture_header.picture_coding_type == picture_coding_type_bidir);
#else
    if (cur_pic->m_picture_header.picture_coding_type == picture_coding_type_bidir || !reordering)
        m_done_pics.push(cur_pic);
    else if (ref_frames[0])
            m_done_pics.push(ref_frames[0]);
#endif
}

void mp2v_decoder_c::decode(uint8_t* buffer, int len) {

    m_bs.set_bitstream_buffer(buffer);

    scan_start_codes(buffer, buffer + len, [&](uint8_t* ptr) {
        if (prev_start_code) {
            last_start_code = ptr;
            uint8_t* p = prev_start_code;
            if (cur_pic) {
                p = cur_pic->bitstream_ptr;
                size_t sz = last_start_code - prev_start_code;
                memcpy(p, prev_start_code, sz);
                cur_pic->bitstream_ptr += sz;
            }
            BITSTREAM((&m_bs));
            bit_idx = 32;
            bit_ptr = (uint32_t*)(p + 4);
            bit_buf = (uint64_t)bswap_32(*((uint32_t*)p));
            uint8_t start_code = *(p + 3);
            switch (start_code) {
            case sequence_header_code: parse_sequence_header(&m_bs, m_sequence_header); break;
            case extension_start_code: decode_extension_data(cur_pic);                  break;
            case group_start_code:     parse_group_of_pictures_header(&m_bs, *(m_group_of_pictures_header = new group_of_pictures_header_t)); break;
            case picture_start_code:
                new_picture = true;
                if (cur_pic) out_pic(cur_pic);
                cur_pic = new_pic();
                parse_picture_header(&m_bs, cur_pic->m_picture_header);
                if (cur_pic->m_picture_header.picture_coding_type == picture_coding_type_pred || cur_pic->m_picture_header.picture_coding_type == picture_coding_type_intra) {
                    cur_pic->add_dependency(ref_frames[1]);
                    ref_frames[0] = ref_frames[1];
                    ref_frames[1] = cur_pic;
                }
                else
                    for (auto* pic : ref_frames) cur_pic->add_dependency(pic);
                break;
            case user_data_start_code: decode_user_data(); break;
            case sequence_error_code:
            case sequence_end_code: break;
            default:
                if ((start_code >= slice_start_code_min) && (start_code <= slice_start_code_max)) {
                    if (new_picture) cur_pic->init();
#ifdef MP2V_MT
                    auto tsk = new mp2v_slice_task_c();
                    tsk->bs = m_bs;
                    cur_pic->add_slice_task(tsk);
#else
                    cur_pic->decode_slice(m_bs);
#endif
                    new_picture = false;
                }
            }
        }
        prev_start_code = ptr;
        });

    if (cur_pic) {
        size_t sz = buffer + len - last_start_code;
        memcpy(cur_pic->bitstream_ptr, prev_start_code, sz);
        cur_pic->bitstream_ptr += sz;
    }
}

void mp2v_slice_task_c::decode() {
    auto pic = (mp2v_picture_c*)owner;
    pic->decode_slice(bs);
}

#ifdef MP2V_MT
void mp2v_decoder_c::threadpool_task_scheduler(mp2v_decoder_c* dec) {
    mp2v_slice_task_c* slice_task = nullptr;
    while (dec->task_queue->get_task((slice_task_c*&)slice_task) == TASK_QUEUE_SUCCESS) {
        slice_task->decode();
        slice_task->done();
    }
}
#endif

void mp2v_decoder_c::decoder_output_scheduler(mp2v_decoder_c* dec) {
#ifdef MP2V_MT
    mp2v_picture_c* pic = nullptr;
    mp2v_picture_c* refs[2] = { 0 };
    while (1) {
        pic = (mp2v_picture_c*)dec->task_queue->get_decoded();
        if (!pic) break;
        if (pic->m_picture_header.picture_coding_type == picture_coding_type_bidir || !dec->reordering) {
            dec->render_func(pic->get_frame());
            pic->render_done();
        }
        else {
            refs[0] = refs[1];
            refs[1] = pic;
            if (refs[0]) {
                dec->render_func(refs[0]->get_frame());
                refs[0]->render_done();
            }
        }
    }
    if (refs[1]) {
        dec->render_func(refs[1]->get_frame());
        refs[1]->render_done();
    }
#else
    mp2v_picture_c* pic = nullptr;
    while (1) {
        dec->m_done_pics.pop(pic);
        if (!pic) break;
        dec->render_func(pic->get_frame());
        dec->m_free_pics.push(pic);
    };
#endif
}

bool mp2v_decoder_c::decoder_init(const decoder_config_t &config, std::function<void(frame_c*)> renderer) {
    int num_pics = config.pictures_pool_size;
    int width = config.width;
    int height = config.height;
    int chroma_format = config.chroma_format;
    reordering = config.reordering;
    render_func = renderer;

#ifdef MP2V_MT
    task_queue = new task_queue_c(num_pics, [&]() -> picture_task_c* {
        return new mp2v_picture_c(this, new frame_c(width, height, chroma_format), config.bitstream_chunk_size);
        });
    for (int i = 0; i < config.num_threads; i++)
        thread_pool[i] = new std::thread(threadpool_task_scheduler, this);
#else
    for (int i = 0; i < num_pics; i++) {
        auto pic = new mp2v_picture_c(this, new frame_c(width, height, chroma_format));
        m_pictures_pool.push_back(pic);
        m_free_pics.push(pic);
    }
#endif

    render_thread = new std::thread(decoder_output_scheduler, this);

    return true;
}

mp2v_decoder_c::~mp2v_decoder_c() {
    if (render_thread && render_thread->joinable()) {
        render_thread->join();
        delete render_thread;
    }
#ifdef MP2V_MT
    for (auto*& thread : thread_pool)
        if (thread && thread->joinable()) {
            thread->join();
            delete thread;
        }
    delete task_queue;
#else
    for (auto* pic : m_pictures_pool) {
        delete pic->get_frame();
        delete pic;
    }
#endif
}