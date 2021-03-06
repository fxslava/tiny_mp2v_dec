// Copyright � 2021 Vladislav Ovchinnikov. All rights reserved.
#include <string.h>
#include "mb_decoder.h"
#include "mp2v_vlc.h"
#include "mc.h"
#include "scan.h"

#if defined(CPU_PLATFORM_AARCH64)
#include "idct_aarch64.hpp"
#elif defined(CPU_PLATFORM_X64)
#include "idct_sse2.hpp"
#else
#include "idct_c.hpp"
#endif

enum mc_template_e {
    mc_templ_field,
    mc_templ_frame
};

template <int chroma_format>
MP2V_INLINE static void inc_macroblock_yuv_ptr(uint8_t* (&yuv)[3]) {
    yuv[0] += 16;
    switch (chroma_format) {
    case chroma_format_420:
        yuv[1] += 8;
        yuv[2] += 8;
        break;
    case chroma_format_422:
        yuv[1] += 8;
        yuv[2] += 8;
        break;
    case chroma_format_444:
        yuv[1] += 16;
        yuv[2] += 16;
        break;
    }
}

template <int chroma_format>
MP2V_INLINE static void inc_macroblock_yuv_ptrs(uint8_t* (&yuv)[3][3]) {
    inc_macroblock_yuv_ptr<chroma_format>(yuv[REF_TYPE_SRC]);
    inc_macroblock_yuv_ptr<chroma_format>(yuv[REF_TYPE_L0]);
    inc_macroblock_yuv_ptr<chroma_format>(yuv[REF_TYPE_L1]);
}
template<bool luma>
MP2V_INLINE int16_t parse_dct_dc_coeff(bitstream_reader_c* bs, uint16_t& dct_dc_pred, int intra_dc_precision) {
    uint16_t dct_dc_differential;
    uint16_t dct_dc_size;
    if (luma) {
        dct_dc_size = get_dct_size_luminance(bs);
        if (dct_dc_size != 0)  dct_dc_differential = bs->read_next_bits(dct_dc_size);
    }
    else {
        dct_dc_size = get_dct_size_chrominance(bs);
        if (dct_dc_size != 0)  dct_dc_differential = bs->read_next_bits(dct_dc_size);
    }

    int16_t dct_diff;
    if (dct_dc_size == 0)
        dct_diff = 0;
    else {
        uint16_t half_range = 1 << (dct_dc_size - 1);
        if (dct_dc_differential >= half_range)
            dct_diff = dct_dc_differential;
        else
            dct_diff = (dct_dc_differential + 1) - (2 * half_range);
    }

    dct_dc_pred += dct_diff;
    return dct_dc_pred << (3 - intra_dc_precision);;
}

template<bool use_dct_one_table, bool intra, bool alt_scan>
static void parse_block(bitstream_reader_c* bs, int16_t* qfs, uint8_t W[64], uint8_t quantizer_scale) {
    int run = 0, level = 0, i = intra ? 1 : 0, sign = 0, sum = 0;
    BITSTREAM(bs);

    if (!use_dct_one_table) {
        UPDATE_BITS();
        uint32_t coef = GET_NEXT_BITS(2);
        if (coef & 2) {
            sign = -(coef & 1);
            int16_t val = (3 * W[i] * quantizer_scale) >> 5;
            sum = qfs[i++] = (val ^ sign) - sign;
            SKIP_BITS(2);
        }
    }

    while (1) {
        UPDATE_BITS();
        uint32_t buffer = GET_NEXT_BITS(32);

        // EOB
        if (use_dct_one_table) { if ((buffer & 0xF0000000u) == (0b0110u << (32u - 4u))) { SKIP_BITS(4); break; } }
        else                   { if ((buffer & 0xc0000000u) == (0b10u << (32u - 2u)))     { SKIP_BITS(2); break; } }

        if ((buffer & 0xfc000000u) == (0b000001u << (32u - 6u))) { // escape code
            buffer <<= 6;
            run = ((uint32_t)buffer >> (32 - 6));
            buffer <<= 6;
            level = ((int32_t)buffer >> (32 - 12));
            sign = ((int32_t)level >> 31); // store sign
            level = (level ^ sign) - sign; // remove sign
            SKIP_BITS(24);
        }
        else if ((buffer & 0xf8000000) == (0b00100 << (32 - 5))) {
            coeff_t coeff = (use_dct_one_table ? vlc_coeff_one_ex : vlc_coeff_zero_ex)[(buffer >> (32 - 8)) & 7];
            level = coeff.level;
            sign = (buffer & (1 << (31 - 8))) ? -1 : 0;
            run = coeff.run;
            SKIP_BITS(9);
        }
        else {
            vlc_lut_coeff_t coeff;
            if (use_dct_one_table) {
                int nlz = bit_scan_reverse(buffer);
                if (nlz > 0) {
                    int idx = buffer >> (32 - nlz - 5);
                    coeff = vlc_coeff_one0[nlz - 1][idx];
                }
                else {
                    nlz = std::min<uint32_t>(bit_scan_reverse(~buffer), 8);
                    int idx = (buffer >> (32 - nlz - 3)) & 7;
                    coeff = vlc_coeff_one1[nlz - 1][idx];
                }
            }
            else {
                int nlz = bit_scan_reverse(buffer);
                int idx = buffer >> (32 - nlz - 5);
                coeff = vlc_coeff_zero[nlz][idx];
            }
            run = coeff.coeff.run;
            sign = (buffer & (1 << (31 - coeff.len))) ? -1 : 0;
            level = coeff.coeff.level;
            SKIP_BITS(coeff.len + 1);
        }

        int32_t val;
        i += run;
        int idx = (int)g_scan_trans[alt_scan ? 1 : 0][i];
        if (intra) val = (level * W[i] * quantizer_scale) >> 4;
        else       val = ((2 * level + 1) * W[i] * quantizer_scale) >> 5;
        val = (val ^ sign) - sign; // apply sign

        sum += (qfs[idx] = std::max<int16_t>(std::min<int16_t>(val, (int16_t)2047), (int16_t)-2048));
        i++;
    }

    sum &= 1;
    sum ^= 1;
    qfs[63] ^= sum;

    UPDATE_BITS();
}

template<bool alt_scan, bool intra, bool add, bool use_dct_one_table, bool luma = false>
MP2V_INLINE void decode_block_template(bitstream_reader_c* m_bs, uint8_t* plane, uint32_t stride, uint8_t W_i[64], uint8_t W[64], uint8_t quantizer_scale, uint16_t& dct_dc_pred, uint8_t intra_dc_prec) {
    ALIGN(32) int16_t QFS[64] = { 0 };
    if (intra) QFS[0] = parse_dct_dc_coeff<luma>(m_bs, dct_dc_pred, intra_dc_prec);
    parse_block<use_dct_one_table, intra, alt_scan>(m_bs, QFS, intra ? W_i : W, quantizer_scale);
    inverse_dct_template<add>(plane, QFS, stride);
}

//decode_transform_template<chroma_format, alt_scan, true, true >(m_bs, cache.yuv_planes[REF_TYPE_SRC], cache.luma_stride, cache.W, coded_block_pattern, cache.quantiser_scale, cache.dct_dc_pred, cache.intra_dc_prec);
template<int chroma_format, bool alt_scan, bool intra, bool add, bool use_dct_one_table>
MP2V_INLINE void decode_transform_template(bitstream_reader_c* m_bs, macroblock_context_cache_t& cache, uint16_t coded_block_pattern, bool dct_type) {
    auto yuv_planes      = cache.yuv_planes[REF_TYPE_SRC];
    auto &dct_dc_pred    = cache.dct_dc_pred;
    auto intra_dc_prec   = cache.intra_dc_prec;
    auto quantizer_scale = cache.quantiser_scale;
    int chroma_stride    = (dct_type && (chroma_format != 1)) ? cache.chroma_stride << 1 : cache.chroma_stride;
    int stride           = dct_type ? cache.luma_stride << 1 : cache.luma_stride;
    auto W               = cache.W;

    // Luma
    if (coded_block_pattern & (1 << 0)) decode_block_template<alt_scan, intra, add, use_dct_one_table, true>(m_bs, yuv_planes[0], stride, W[0], W[1], quantizer_scale, dct_dc_pred[0], intra_dc_prec);
    if (coded_block_pattern & (1 << 1)) decode_block_template<alt_scan, intra, add, use_dct_one_table, true>(m_bs, yuv_planes[0] + 8, stride, W[0], W[1], quantizer_scale, dct_dc_pred[0], intra_dc_prec);
    if (coded_block_pattern & (1 << 2)) decode_block_template<alt_scan, intra, add, use_dct_one_table, true>(m_bs, yuv_planes[0] + (dct_type ? cache.luma_stride : 8 * stride), stride, W[0], W[1], quantizer_scale, dct_dc_pred[0], intra_dc_prec);
    if (coded_block_pattern & (1 << 3)) decode_block_template<alt_scan, intra, add, use_dct_one_table, true>(m_bs, yuv_planes[0] + (dct_type ? cache.luma_stride : 8 * stride) + 8, stride, W[0], W[1], quantizer_scale, dct_dc_pred[0], intra_dc_prec);

    // Chroma format 4:2:0
    if (chroma_format >= 1) {
        if (coded_block_pattern & (1 << 4)) decode_block_template<alt_scan, intra, add, use_dct_one_table>(m_bs, yuv_planes[1], chroma_stride, W[0], W[1], quantizer_scale, dct_dc_pred[1], intra_dc_prec);
        if (coded_block_pattern & (1 << 5)) decode_block_template<alt_scan, intra, add, use_dct_one_table>(m_bs, yuv_planes[2], chroma_stride, W[0], W[1], quantizer_scale, dct_dc_pred[2], intra_dc_prec); }
    // Chroma format 4:2:2
    if (chroma_format >= 2) {
        if (coded_block_pattern & (1 << 6)) decode_block_template<alt_scan, intra, add, use_dct_one_table>(m_bs, yuv_planes[1] + (dct_type ? cache.chroma_stride : 8 * chroma_stride), chroma_stride, W[2], W[3], quantizer_scale, dct_dc_pred[1], intra_dc_prec);
        if (coded_block_pattern & (1 << 7)) decode_block_template<alt_scan, intra, add, use_dct_one_table>(m_bs, yuv_planes[2] + (dct_type ? cache.chroma_stride : 8 * chroma_stride), chroma_stride, W[2], W[3], quantizer_scale, dct_dc_pred[2], intra_dc_prec); }
    // Chroma format 4:4:4
    if (chroma_format == 3) {
        if (coded_block_pattern & (1 << 8))  decode_block_template<alt_scan, intra, add, use_dct_one_table>(m_bs, yuv_planes[1] + 8, chroma_stride, W[2], W[3], quantizer_scale, dct_dc_pred[1], intra_dc_prec);
        if (coded_block_pattern & (1 << 9))  decode_block_template<alt_scan, intra, add, use_dct_one_table>(m_bs, yuv_planes[2] + 8, chroma_stride, W[2], W[3], quantizer_scale, dct_dc_pred[2], intra_dc_prec);
        if (coded_block_pattern & (1 << 10)) decode_block_template<alt_scan, intra, add, use_dct_one_table>(m_bs, yuv_planes[1] + (dct_type ? 1 : 8) * stride + 8, chroma_stride, W[2], W[3], quantizer_scale, dct_dc_pred[1], intra_dc_prec);
        if (coded_block_pattern & (1 << 11)) decode_block_template<alt_scan, intra, add, use_dct_one_table>(m_bs, yuv_planes[2] + (dct_type ? 1 : 8) * stride + 8, chroma_stride, W[2], W[3], quantizer_scale, dct_dc_pred[2], intra_dc_prec); }
}

template<int chroma_format, int plane_idx>
MP2V_INLINE void apply_chroma_scale(int16_t& mvx, int16_t& mvy) {
    if (plane_idx > 0) {
        if (chroma_format < 3)
            mvx >>= 1;
        if (chroma_format < 2)
            mvy >>= 1;
    }
}

MP2V_INLINE int mc_bidir_idx(int16_t mvfx, int16_t mvfy, int16_t mvbx, int16_t mvby) {
    return (mvfx & 0x01) + ((mvfy & 0x01) << 1) + ((mvbx & 0x01) << 2) + ((mvby & 0x01) << 3);
}

template<int chroma_format, int plane_idx, int vect_idx, mc_template_e mc_templ>
MP2V_INLINE void mc_bidir_template(uint8_t* dst, uint8_t* ref0, uint8_t* ref1, macroblock_t &mb, uint32_t stride, uint32_t chroma_stride, int16_t MVs[2][2][2]) {
    auto  _stride = (mc_templ == mc_templ_field) ? stride << 1 : stride;
    auto  _chroma_stride = (mc_templ == mc_templ_field) ? chroma_stride << 1 : chroma_stride;
    uint8_t* fref = ref0;
    uint8_t* bref = ref1;
    auto  mvfx = MVs[vect_idx][0][0];
    auto  mvfy = MVs[vect_idx][0][1];
    auto  mvbx = MVs[vect_idx][1][0];
    auto  mvby = MVs[vect_idx][1][1];
    apply_chroma_scale<chroma_format, plane_idx>(mvfx, mvfy);
    apply_chroma_scale<chroma_format, plane_idx>(mvbx, mvby);
    int mvs_ridx = mc_bidir_idx(mvfx, mvfy, mvbx, mvby);
    fref += static_cast<ptrdiff_t>(mvfx >> 1) + static_cast<ptrdiff_t>(mvfy >> 1) * (plane_idx ? _chroma_stride : _stride);
    bref += static_cast<ptrdiff_t>(mvbx >> 1) + static_cast<ptrdiff_t>(mvby >> 1) * (plane_idx ? _chroma_stride : _stride);

    auto plane_stride = (plane_idx == 0) ? stride : chroma_stride;
    if (mc_templ == mc_templ_field) {
        if (mb.motion_vertical_field_select[vect_idx][0])
            fref += plane_stride;
        if (mb.motion_vertical_field_select[vect_idx][1])
            bref += plane_stride;
        if (vect_idx)
            dst += plane_stride;
    }

    if (plane_idx == 0) {
        switch (mc_templ) {
        case mc_templ_field: mc_bidir_16xh[mvs_ridx](dst, bref, fref, _stride, 8); break;
        case mc_templ_frame: mc_bidir_16xh[mvs_ridx](dst, bref, fref, _stride, 16); break;
        }
    }
    else {
        switch (chroma_format) {
        case chroma_format_420: mc_bidir_8xh[mvs_ridx](dst, bref, fref, _chroma_stride, (mc_templ == mc_templ_field) ? 4 : 8); break;
        case chroma_format_422: mc_bidir_8xh[mvs_ridx](dst, bref, fref, _chroma_stride, (mc_templ == mc_templ_field) ? 8 : 16); break;
        case chroma_format_444: mc_bidir_16xh[mvs_ridx](dst, bref, fref, _chroma_stride, (mc_templ == mc_templ_field) ? 8 : 16); break;
        }
    }
}

MP2V_INLINE int mc_unidir_idx(int16_t mvx, int16_t mvy) {
    return (mvx & 0x01) | ((mvy & 0x01) << 1);
}

template<int chroma_format, int plane_idx, int vect_idx, mc_template_e mc_templ, bool forward>
MP2V_INLINE void mc_unidir_template(uint8_t* dst, uint8_t* ref, macroblock_t &mb, uint32_t stride, uint32_t chroma_stride, int16_t MVs[2][2][2]) {
    auto  _stride = (mc_templ == mc_templ_field) ? stride << 1 : stride;
    auto  _chroma_stride = (mc_templ == mc_templ_field) ? chroma_stride << 1 : chroma_stride;
    auto  mvx = MVs[vect_idx][forward ? 0 : 1][0];
    auto  mvy = MVs[vect_idx][forward ? 0 : 1][1];
    apply_chroma_scale<chroma_format, plane_idx>(mvx, mvy);
    int mvs_ridx = mc_unidir_idx(mvx, mvy);
    int offset = (mvx >> 1) + (mvy >> 1) * (plane_idx ? _chroma_stride : _stride);
    ref += static_cast<ptrdiff_t>(offset);

    auto plane_stride = (plane_idx == 0) ? stride : chroma_stride;
    if (mc_templ == mc_templ_field) {
        if (mb.motion_vertical_field_select[vect_idx][forward ? 0 : 1])
            ref += plane_stride;
        if (vect_idx)
            dst += plane_stride;
    }

    if (plane_idx == 0) {
        switch (mc_templ) {
        case mc_templ_field: mc_pred_16xh[mvs_ridx](dst, ref, _stride,  8); break;
        case mc_templ_frame: mc_pred_16xh[mvs_ridx](dst, ref, _stride, 16); break;
        }
    }
    else {
        switch (chroma_format) {
        case chroma_format_420: mc_pred_8xh[mvs_ridx](dst, ref, _chroma_stride, (mc_templ == mc_templ_field) ? 4 :  8); break;
        case chroma_format_422: mc_pred_8xh[mvs_ridx](dst, ref, _chroma_stride, (mc_templ == mc_templ_field) ? 8 : 16); break;
        case chroma_format_444: mc_pred_16xh[mvs_ridx](dst, ref, _chroma_stride, (mc_templ == mc_templ_field) ? 8 : 16); break;
        }
    }
}

template<int chroma_format, mc_template_e mc_templ, bool two_vect, bool skipped = false>
MP2V_INLINE void base_motion_compensation(macroblock_context_cache_t& cache, macroblock_t &mb, int16_t MVs[2][2][2]) {
    auto dst = cache.yuv_planes[REF_TYPE_SRC];
    auto ref0 = cache.yuv_planes[REF_TYPE_L0];
    auto ref1 = cache.yuv_planes[REF_TYPE_L1];
    auto stride = cache.luma_stride;
    auto chroma_stride = cache.chroma_stride;
    auto macroblock_type = skipped ? cache.previous_mb_type : mb.macroblock_type;

    if ((macroblock_type & macroblock_motion_forward_bit) && (macroblock_type & macroblock_motion_backward_bit)) {
        mc_bidir_template<chroma_format, 0, 0, mc_templ>(dst[0], ref0[0], ref1[0], mb, stride, chroma_stride, MVs);
        mc_bidir_template<chroma_format, 1, 0, mc_templ>(dst[1], ref0[1], ref1[1], mb, stride, chroma_stride, MVs);
        mc_bidir_template<chroma_format, 2, 0, mc_templ>(dst[2], ref0[2], ref1[2], mb, stride, chroma_stride, MVs);
        if (two_vect) {
            mc_bidir_template<chroma_format, 0, 1, mc_templ>(dst[0], ref0[0], ref1[0], mb, stride, chroma_stride, MVs);
            mc_bidir_template<chroma_format, 1, 1, mc_templ>(dst[1], ref0[1], ref1[1], mb, stride, chroma_stride, MVs);
            mc_bidir_template<chroma_format, 2, 1, mc_templ>(dst[2], ref0[2], ref1[2], mb, stride, chroma_stride, MVs);
        }
    } else
    if ((macroblock_type & macroblock_motion_forward_bit) && !(macroblock_type & macroblock_motion_backward_bit)) {
        mc_unidir_template<chroma_format, 0, 0, mc_templ, true>(dst[0], ref0[0], mb, stride, chroma_stride, MVs);
        mc_unidir_template<chroma_format, 1, 0, mc_templ, true>(dst[1], ref0[1], mb, stride, chroma_stride, MVs);
        mc_unidir_template<chroma_format, 2, 0, mc_templ, true>(dst[2], ref0[2], mb, stride, chroma_stride, MVs);
        if (two_vect) {
            mc_unidir_template<chroma_format, 0, 1, mc_templ, true>(dst[0], ref0[0], mb, stride, chroma_stride, MVs);
            mc_unidir_template<chroma_format, 1, 1, mc_templ, true>(dst[1], ref0[1], mb, stride, chroma_stride, MVs);
            mc_unidir_template<chroma_format, 2, 1, mc_templ, true>(dst[2], ref0[2], mb, stride, chroma_stride, MVs);
        }
    } else
    if (!(macroblock_type & macroblock_motion_forward_bit) && (macroblock_type & macroblock_motion_backward_bit)) {
        mc_unidir_template<chroma_format, 0, 0, mc_templ, false>(dst[0], ref1[0], mb, stride, chroma_stride, MVs);
        mc_unidir_template<chroma_format, 1, 0, mc_templ, false>(dst[1], ref1[1], mb, stride, chroma_stride, MVs);
        mc_unidir_template<chroma_format, 2, 0, mc_templ, false>(dst[2], ref1[2], mb, stride, chroma_stride, MVs);
        if (two_vect) {
            mc_unidir_template<chroma_format, 0, 1, mc_templ, false>(dst[0], ref1[0], mb, stride, chroma_stride, MVs);
            mc_unidir_template<chroma_format, 1, 1, mc_templ, false>(dst[1], ref1[1], mb, stride, chroma_stride, MVs);
            mc_unidir_template<chroma_format, 2, 1, mc_templ, false>(dst[2], ref1[2], mb, stride, chroma_stride, MVs);
        }
    } else {
        mc_unidir_template<chroma_format, 0, 0, mc_templ, true>(dst[0], ref0[0], mb, stride, chroma_stride, MVs);
        mc_unidir_template<chroma_format, 1, 0, mc_templ, true>(dst[1], ref0[1], mb, stride, chroma_stride, MVs);
        mc_unidir_template<chroma_format, 2, 0, mc_templ, true>(dst[2], ref0[2], mb, stride, chroma_stride, MVs);
        if (two_vect) {
            mc_unidir_template<chroma_format, 0, 1, mc_templ, true>(dst[0], ref0[0], mb, stride, chroma_stride, MVs);
            mc_unidir_template<chroma_format, 1, 1, mc_templ, true>(dst[1], ref0[1], mb, stride, chroma_stride, MVs);
            mc_unidir_template<chroma_format, 2, 1, mc_templ, true>(dst[2], ref0[2], mb, stride, chroma_stride, MVs);
        }
    }
}

template<int picture_coding_type, int picture_structure, int frame_pred_frame_dct>
static bool parse_modes(bitstream_reader_c* m_bs, macroblock_t& mb, int spatial_temporal_weight_code_table_index, mv_format_e& mv_format) {
    mb.macroblock_type = get_macroblock_type(m_bs, picture_coding_type);
    if ((mb.macroblock_type & spatial_temporal_weight_code_flag_bit) && (spatial_temporal_weight_code_table_index != 0)) {
        /*uint32_t spatial_temporal_weight_code = */m_bs->read_next_bits(2);
    }
    if ((mb.macroblock_type & macroblock_motion_forward_bit) || (mb.macroblock_type & macroblock_motion_backward_bit)) {
        if (picture_structure == picture_structure_framepic) {
            mb.frame_motion_type = 2; // Frame-based
            if (frame_pred_frame_dct == 0)
                mb.frame_motion_type = m_bs->read_next_bits(2);
        }
        else {
            mb.field_motion_type = m_bs->read_next_bits(2);
        }
    }
    if (frame_pred_frame_dct) mb.dct_type = 0;
    if ((picture_structure == picture_structure_framepic) && (frame_pred_frame_dct == 0) &&
        ((mb.macroblock_type & macroblock_intra_bit) || (mb.macroblock_type & macroblock_pattern_bit))) {
        mb.dct_type = m_bs->read_next_bits(1);
    }

    // decode modes
    if (mb.macroblock_type & macroblock_intra_bit)
    {
        mb.motion_vector_count = 0;
        mb.dmv = 0;
        if (picture_structure == picture_structure_framepic) {
            mv_format = Frame;
            mb.prediction_type = Frame_based;
        }
        else {
            mv_format = Field;
            mb.prediction_type = Field_based;
        }
    }
    else
    {
        mb.motion_vector_count = 1;
        mb.dmv = 0;
        if (picture_structure == picture_structure_framepic) {
            switch (mb.frame_motion_type) {
            case 1:
                mv_format = Field;
                mb.motion_vector_count = 2;
                mb.prediction_type = Field_based;
                break;
            case 2:
                mv_format = Frame;
                mb.prediction_type = Frame_based;
                break;
            case 3:
                mv_format = Field;
                mb.prediction_type = Dual_Prime;
                mb.dmv = 1;
                break;
            }
        }
        else {
            switch (mb.field_motion_type) {
            case 1:
                mv_format = Field;
                mb.prediction_type = Field_based;
                break;
            case 2:
                mv_format = Field;
                mb.motion_vector_count = 2;
                mb.prediction_type = MC16x8;
                break;
            case 3:
                mv_format = Field;
                mb.prediction_type = Dual_Prime;
                mb.dmv = 1;
                break;
            }
        }
    }
    return true;
}

template<int chroma_format>
MP2V_INLINE uint16_t parse_coded_block_pattern(bitstream_reader_c* m_bs, macroblock_t& mb) {
    uint16_t coded_block_pattern = 0;
    uint32_t coded_block_pattern_1, coded_block_pattern_2, cbp;

    cbp = get_coded_block_pattern(m_bs);
    if (chroma_format == chroma_format_422)
        coded_block_pattern_1 = m_bs->read_next_bits(2);
    if (chroma_format == chroma_format_444)
        coded_block_pattern_2 = m_bs->read_next_bits(6);
    
    if (mb.macroblock_type & macroblock_intra_bit)
        coded_block_pattern = 0xffff;
    if (mb.macroblock_type & macroblock_pattern_bit) {
        for (int i = 0; i < 6; i++)
            if (cbp & (1 << (5 - i))) coded_block_pattern |= (1 << i);
        if (chroma_format == chroma_format_422)
            for (int i = 6; i < 8; i++)
                if (coded_block_pattern_1 & (1 << (7 - i))) coded_block_pattern |= (1 << i);
        if (chroma_format == chroma_format_444)
            for (int i = 6; i < 12; i++)
                if (coded_block_pattern_2 & (1 << (11 - i))) coded_block_pattern |= (1 << i);
    }
    return coded_block_pattern;
}

template<uint8_t picture_structure, int t, bool residual>
MP2V_INLINE void update_motion_predictor(uint32_t f_code, int32_t motion_code, uint32_t motion_residual, int16_t& PMV, int16_t& MVs, mv_format_e mv_format) {
    int r_size = f_code - 1;
    int f = 1 << r_size;
    int high = (16 * f) - 1;
    int low = ((-16) * f);
    int range = (32 * f);

    int delta;
    if (residual) {
        delta = ((labs(motion_code) - 1) * f) + motion_residual + 1;
        if (motion_code < 0)
            delta = -delta;
    }
    else
        delta = motion_code;

    int prediction = PMV;
    if ((mv_format == Field) && (t == 1) && (picture_structure == picture_structure_framepic))
        prediction = PMV >> 1;

    MVs = prediction + delta;

    if (MVs < low)  MVs += range;
    if (MVs > high) MVs -= range;

    if ((mv_format == Field) && (t == 1) && (picture_structure == picture_structure_framepic))
        PMV = MVs * 2;
    else
        PMV = MVs;
}

template<uint8_t picture_structure, bool dmv>
MP2V_INLINE bool parse_motion_vector(bitstream_reader_c* m_bs, uint32_t f_code[2], int16_t PMV[2], int16_t MVs[2], mv_format_e mv_format) {
    int32_t motion_code = get_motion_code(m_bs);
    if ((f_code[0] != 1) && (motion_code != 0)) {
        uint32_t motion_residual = m_bs->read_next_bits(f_code[0] - 1);
        update_motion_predictor<picture_structure, 0, true>(f_code[0], motion_code, motion_residual, PMV[0], MVs[0], mv_format);
    }
    else
        update_motion_predictor<picture_structure, 0, false>(f_code[0], motion_code, 0, PMV[0], MVs[0], mv_format);

    if (dmv)
        /*mb.dmvector[0] = */get_dmvector(m_bs);

    motion_code = get_motion_code(m_bs);
    if ((f_code[1] != 1) && (motion_code != 0)) {
        uint32_t motion_residual = m_bs->read_next_bits(f_code[1] - 1);
        update_motion_predictor<picture_structure, 1, true>(f_code[1], motion_code, motion_residual, PMV[1], MVs[1], mv_format);
    }
    else
        update_motion_predictor<picture_structure, 1, false>(f_code[1], motion_code, 0, PMV[1], MVs[1], mv_format);

    if (dmv)
        /*mb.dmvector[1] = */get_dmvector(m_bs);
    return true;
}

template <uint8_t picture_structure, int s, bool dmv>
MP2V_INLINE bool parse_motion_vectors(bitstream_reader_c* m_bs, macroblock_t& mb, uint32_t f_code[2][2], int16_t PMV[2][2][2], int16_t MVs[2][2][2], mv_format_e mv_format) {
    if (mb.motion_vector_count == 1) {
        if ((mv_format == Field) && !dmv)
            mb.motion_vertical_field_select[0][s] = m_bs->read_next_bits(1);
        parse_motion_vector<picture_structure, dmv>(m_bs, f_code[s], PMV[0][s], MVs[0][s], mv_format);
    }
    else {
        mb.motion_vertical_field_select[0][s] = m_bs->read_next_bits(1);
        parse_motion_vector<picture_structure, dmv>(m_bs, f_code[s], PMV[0][s], MVs[0][s], mv_format);
        mb.motion_vertical_field_select[1][s] = m_bs->read_next_bits(1);
        parse_motion_vector<picture_structure, dmv>(m_bs, f_code[s], PMV[1][s], MVs[1][s], mv_format);
    }
    return true;
}

template<uint8_t picture_coding_type,        //3 bit (I, P, B)
         uint8_t picture_structure,          //2 bit (top|bottom field, frame)
         uint8_t frame_pred_frame_dct,       //1 bit // only with picture_structure == frame
         uint8_t concealment_motion_vectors, //1 bit // only with picture_coding_type == I
         uint8_t chroma_format,              //2 bit (420, 422, 444)
         bool q_scale_type, bool alt_scan>
bool parse_macroblock_template(bitstream_reader_c* m_bs, macroblock_context_cache_t &cache) {
#ifdef _DEBUG
    auto& mb = cache.mb;
#else
    macroblock_t mb;
#endif

    mb.macroblock_address_increment = 0;
    while (m_bs->get_next_bits(vlc_macroblock_escape_code.len) == vlc_macroblock_escape_code.value) {
        m_bs->skip_bits(vlc_macroblock_escape_code.len);
        mb.macroblock_address_increment += 33;
    }
    mb.macroblock_address_increment += get_macroblock_address_increment(m_bs);

    // decode skipped macroblocks
    if ((mb.macroblock_address_increment > 1) && (picture_coding_type == picture_coding_type_pred))
        memset(cache.PMVs, 0, sizeof(cache.PMVs));
    for (int i = 0; i < (int)mb.macroblock_address_increment; i++) {
        if ((uint32_t)i == (mb.macroblock_address_increment - 1)) break;
        if (picture_structure == picture_structure_framepic) {
            if (picture_coding_type == picture_coding_type_bidir) base_motion_compensation<chroma_format, mc_templ_frame, true,  true>(cache, mb, cache.PMVs);
            else                                                  base_motion_compensation<chroma_format, mc_templ_frame, false, true>(cache, mb, cache.PMVs); }
        inc_macroblock_yuv_ptrs<chroma_format>(cache.yuv_planes);
    }

    // Parse Macroblock Modes
    mv_format_e mv_format;
    parse_modes<picture_coding_type, picture_structure, frame_pred_frame_dct>(m_bs, mb, 0, mv_format);
    if (mb.macroblock_type & macroblock_quant_bit) {
        auto quantiser_scale_code = m_bs->read_next_bits(5);
        if (q_scale_type) {
            if (quantiser_scale_code < 9)       cache.quantiser_scale =  quantiser_scale_code;
            else if (quantiser_scale_code < 17) cache.quantiser_scale = (quantiser_scale_code -  4) << 1;
            else if (quantiser_scale_code < 25) cache.quantiser_scale = (quantiser_scale_code - 10) << 2;
            else                                cache.quantiser_scale = (quantiser_scale_code - 17) << 3;
        }   else                                cache.quantiser_scale =  quantiser_scale_code << 1;
    }

    // Parse Motion Vectors
    int16_t  MVs[2][2][2];
    if ((mb.macroblock_type & macroblock_motion_forward_bit) || ((mb.macroblock_type & macroblock_intra_bit) && concealment_motion_vectors)) {
        if (mb.dmv) parse_motion_vectors<picture_structure, 0, true> (m_bs, mb, cache.f_code, cache.PMVs, MVs, mv_format);
        else        parse_motion_vectors<picture_structure, 0, false>(m_bs, mb, cache.f_code, cache.PMVs, MVs, mv_format); }
    if ((mb.macroblock_type & macroblock_motion_backward_bit) != 0) {
        if (mb.dmv) parse_motion_vectors<picture_structure, 1, true> (m_bs, mb, cache.f_code, cache.PMVs, MVs, mv_format);
        else        parse_motion_vectors<picture_structure, 1, false>(m_bs, mb, cache.f_code, cache.PMVs, MVs, mv_format); }
    if (((mb.macroblock_type & macroblock_intra_bit) != 0) && concealment_motion_vectors)
        m_bs->skip_bits(1);

#ifdef _DEBUG
    memcpy(mb.MVs, MVs, sizeof(MVs));
#endif

    // Update motion vectors predictors conditions (Table 7-9 � Updating of motion vector predictors in frame pictures)
    if (picture_coding_type != picture_coding_type_intra) {
        if (mb.prediction_type == Frame_based) {
            if (mb.macroblock_type & macroblock_intra_bit)
                for (int t : { 0, 1 }) cache.PMVs[1][0][t] = cache.PMVs[0][0][t];
            if ((mb.macroblock_type & macroblock_motion_forward_bit) && (mb.macroblock_type & macroblock_motion_backward_bit) && !(mb.macroblock_type & macroblock_intra_bit))
                for (int t : { 0, 1 }) {
                    cache.PMVs[1][0][t] = cache.PMVs[0][0][t];
                    cache.PMVs[1][1][t] = cache.PMVs[0][1][t];
                }
            if ((mb.macroblock_type & macroblock_motion_forward_bit) && !(mb.macroblock_type & macroblock_motion_backward_bit) && !(mb.macroblock_type & macroblock_intra_bit))
                for (int t : { 0, 1 }) cache.PMVs[1][0][t] = cache.PMVs[0][0][t];
            if (!(mb.macroblock_type & macroblock_motion_forward_bit) && (mb.macroblock_type & macroblock_motion_backward_bit) && !(mb.macroblock_type & macroblock_intra_bit))
                for (int t : { 0, 1 }) cache.PMVs[1][1][t] = cache.PMVs[0][1][t];
        }
        if (mb.prediction_type == Dual_Prime)
            if ((mb.macroblock_type & macroblock_motion_forward_bit) && !(mb.macroblock_type & macroblock_motion_backward_bit) && !(mb.macroblock_type & macroblock_intra_bit))
                for (int t : { 0, 1 }) cache.PMVs[1][0][t] = cache.PMVs[0][0][t];

        if (((mb.macroblock_type & macroblock_intra_bit) && !concealment_motion_vectors) ||
            ((picture_coding_type == picture_coding_type_pred) && !(mb.macroblock_type & macroblock_intra_bit) && !(mb.macroblock_type & macroblock_motion_forward_bit))) {
            memset(cache.PMVs, 0, sizeof(cache.PMVs));
            memset(MVs, 0, sizeof(MVs));
            mb.prediction_type = (picture_structure == picture_structure_framepic) ? Frame_based : Field_based;
        }

        // Motion compensation
        if (!(mb.macroblock_type & macroblock_intra_bit)) {
            switch (mb.prediction_type) {
            case Field_based:
                if (mb.motion_vector_count == 2) base_motion_compensation<chroma_format, mc_templ_field, true>(cache, mb, MVs);
                else                             base_motion_compensation<chroma_format, mc_templ_field, false>(cache, mb, MVs); 
                break;
            case Frame_based:
                if (mb.motion_vector_count == 2) base_motion_compensation<chroma_format, mc_templ_frame, true>(cache, mb, MVs);
                else                             base_motion_compensation<chroma_format, mc_templ_frame, false>(cache, mb, MVs); 
                break;
            case Dual_Prime:
            case MC16x8: break; // Not supported
            }
        }
    }

    bool intra_block = mb.macroblock_type & macroblock_intra_bit;
    if ((mb.macroblock_address_increment > 1) || !intra_block)
        for (auto& pred : cache.dct_dc_pred) 
            pred = 1 << (cache.intra_dc_prec + 7);

    bool dct_one_table = (cache.intra_vlc_format == 1) && intra_block;
    uint16_t coded_block_pattern = 0xffff;
    if (mb.macroblock_type & macroblock_pattern_bit)
        coded_block_pattern = parse_coded_block_pattern<chroma_format>(m_bs, mb);
    if ((mb.macroblock_type & macroblock_pattern_bit) || intra_block){
        if (dct_one_table)    decode_transform_template<chroma_format, alt_scan, true, false, true >(m_bs, cache, coded_block_pattern, mb.dct_type);
        else if (intra_block) decode_transform_template<chroma_format, alt_scan, true, false, false>(m_bs, cache, coded_block_pattern, mb.dct_type);
        else                  decode_transform_template<chroma_format, alt_scan, false, true, false>(m_bs, cache, coded_block_pattern, mb.dct_type);
    }

    inc_macroblock_yuv_ptrs<chroma_format>(cache.yuv_planes);
    cache.previous_mb_type = mb.macroblock_type;
    return true;
}

#define SEL_CHROMA_FROMATS_PARSE_MACROBLOCKS_ROUTINES(pct, ps, fpfdct, cmv) { \
    switch (chroma_format) { \
    case chroma_format_420:  \
        if (!q_scale_type) { \
               if (!alt_scan) return parse_macroblock_template<pct, ps, fpfdct, cmv, chroma_format_420, false, false>;    \
               else           return parse_macroblock_template<pct, ps, fpfdct, cmv, chroma_format_420, false, true>; }   \
        else { if (!alt_scan) return parse_macroblock_template<pct, ps, fpfdct, cmv, chroma_format_420, true, false>;     \
               else           return parse_macroblock_template<pct, ps, fpfdct, cmv, chroma_format_420, true, true>; }    \
    case chroma_format_422:                                                                                               \
        if (!q_scale_type) {                                                                                              \
               if (!alt_scan) return parse_macroblock_template<pct, ps, fpfdct, cmv, chroma_format_422, false, false>;    \
               else           return parse_macroblock_template<pct, ps, fpfdct, cmv, chroma_format_422, false, true>; }   \
        else { if (!alt_scan) return parse_macroblock_template<pct, ps, fpfdct, cmv, chroma_format_422, true, false>;     \
               else           return parse_macroblock_template<pct, ps, fpfdct, cmv, chroma_format_422, true, true>; }    \
    case chroma_format_444:                                                                                               \
        if (!q_scale_type) {                                                                                              \
               if (!alt_scan) return parse_macroblock_template<pct, ps, fpfdct, cmv, chroma_format_444, false, false>;    \
               else           return parse_macroblock_template<pct, ps, fpfdct, cmv, chroma_format_444, false, true>; }   \
        else { if (!alt_scan) return parse_macroblock_template<pct, ps, fpfdct, cmv, chroma_format_444, true, false>;     \
               else           return parse_macroblock_template<pct, ps, fpfdct, cmv, chroma_format_444, true, true>; } } }

#define SEL_FRAME_FIELD_PARSE_MACROBLOCKS_ROUTINES(pct, cmv) \
    if (picture_structure == picture_structure_framepic) { \
        if (frame_pred_frame_dct) \
            SEL_CHROMA_FROMATS_PARSE_MACROBLOCKS_ROUTINES(pct, picture_structure_framepic, 1, cmv) \
        else \
            SEL_CHROMA_FROMATS_PARSE_MACROBLOCKS_ROUTINES(pct, picture_structure_framepic, 0, cmv) \
    } else \
        SEL_CHROMA_FROMATS_PARSE_MACROBLOCKS_ROUTINES(pct, picture_structure_topfield, 0, cmv)

parse_macroblock_func_t select_parse_macroblock_func(uint8_t picture_coding_type, uint8_t picture_structure, uint8_t frame_pred_frame_dct, uint8_t concealment_motion_vectors, uint8_t chroma_format, bool q_scale_type, bool alt_scan)
{
    switch (picture_coding_type) {
    case picture_coding_type_intra:
        if (concealment_motion_vectors) {
            SEL_FRAME_FIELD_PARSE_MACROBLOCKS_ROUTINES(picture_coding_type_intra, 1)
        }
        else {
            SEL_FRAME_FIELD_PARSE_MACROBLOCKS_ROUTINES(picture_coding_type_intra, 0)
        }
    case picture_coding_type_pred:
        SEL_FRAME_FIELD_PARSE_MACROBLOCKS_ROUTINES(picture_coding_type_pred, 0)
    case picture_coding_type_bidir:
        SEL_FRAME_FIELD_PARSE_MACROBLOCKS_ROUTINES(picture_coding_type_bidir, 0)
    default:
        return 0;
    };
};
