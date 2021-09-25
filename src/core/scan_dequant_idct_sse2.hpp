// Copyright © 2021 Vladislav Ovchinnikov. All rights reserved.
#pragma once
#include <stdint.h>
#include <emmintrin.h>
#include "common/cpu.hpp"
#include "bitstream.h"
#include "scan_sse4.hpp"
#include "idct_sse2.hpp"

__m128i insert_epi16(__m128i reg, int value, int pos) {
    __m128i tmp = _mm_cvtsi32_si128((pos & 1) ? value : value << 16);
    if (pos & 2)
        tmp = _mm_bslli_si128(tmp, 2);
    if (pos & 4)
        tmp = _mm_bslli_si128(tmp, 4);
    return _mm_or_si128(reg, tmp);
}

template<bool use_dct_one_table>
MP2V_INLINE bool read_coefficient(__m128i& reg, __m128i& next_reg, uint32_t*& bit_ptr, uint64_t& bit_buf, uint32_t& bit_idx, int& n)
{
    int run = 0, signed_level = 0;

    while (1) {
        UPDATE_BITS();
        uint32_t buffer = GET_NEXT_BITS(32);

        // EOB
        if (use_dct_one_table) { if ((buffer & 0xF0000000) == (0b0110ll << (32 - 4))) { SKIP_BITS(4); return true; } }
        else { if ((buffer & 0xc0000000) == (0b10 << (32 - 2))) { SKIP_BITS(2); return true; } }

        if ((buffer & 0xfc000000) == (0b000001 << (32 - 6))) { // escape code
            run = (buffer >> (32 - 12)) & 0x3f;
            signed_level = (buffer >> (32 - 24)) & 0xfff;
            if (signed_level & 0b100000000000)
                signed_level |= 0xfffff000;
            SKIP_BITS(24);
        }
        else if ((buffer & 0xf8000000) == (0b00100 << (32 - 5))) {
            coeff_t coeff = use_dct_one_table ? vlc_coeff_one_ex[(buffer >> (32 - 8)) & 7] : vlc_coeff_zero_ex[(buffer >> (32 - 8)) & 7];
            signed_level = (buffer & (0b000000001 << (31 - 9))) ? -coeff.level : coeff.level;
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
            signed_level = (buffer & (1 << (31 - coeff.len))) ? -coeff.coeff.level : coeff.coeff.level;
            SKIP_BITS(coeff.len + 1);
        }

        n += run;
        if (n >= 8) {
            n -= 8;
            break;
        }
        reg = insert_epi16(reg, signed_level, n++);
    }
    next_reg = insert_epi16(next_reg, signed_level, n++);
    return false;
}

template<bool use_dct_one_table, bool intra>
static void parse_block(__m128i buffer[8], bitstream_reader_c* bs) {
    int run = 0, signed_level = 0, n = intra ? 1 : 0;
    BITSTREAM(bs);

    if (!use_dct_one_table) {
        UPDATE_BITS();
        uint32_t coef = GET_NEXT_BITS(2);
        if (coef == 2) {
            buffer[0] = insert_epi16(buffer[0], +1, n++);
            SKIP_BITS(2); 
        }
        if (coef == 3) { 
            buffer[0] = insert_epi16(buffer[0], -1, n++);
            SKIP_BITS(2); 
        }
    }

    while (1) {
        if (read_coefficient<use_dct_one_table>(buffer[0], buffer[1], bit_ptr, bit_buf, bit_idx, n)) break;
        if (read_coefficient<use_dct_one_table>(buffer[1], buffer[2], bit_ptr, bit_buf, bit_idx, n)) break;
        if (read_coefficient<use_dct_one_table>(buffer[2], buffer[3], bit_ptr, bit_buf, bit_idx, n)) break;
        if (read_coefficient<use_dct_one_table>(buffer[3], buffer[4], bit_ptr, bit_buf, bit_idx, n)) break;
        if (read_coefficient<use_dct_one_table>(buffer[4], buffer[5], bit_ptr, bit_buf, bit_idx, n)) break;
        if (read_coefficient<use_dct_one_table>(buffer[5], buffer[6], bit_ptr, bit_buf, bit_idx, n)) break;
        if (read_coefficient<use_dct_one_table>(buffer[6], buffer[7], bit_ptr, bit_buf, bit_idx, n)) break;
        if (read_coefficient<use_dct_one_table>(buffer[7], buffer[7], bit_ptr, bit_buf, bit_idx, n)) break;
    }

    UPDATE_BITS();
}

template<bool intra, bool add, bool use_dct_one_table>
void scan_dequant_idct_template_sse2(bitstream_reader_c* m_bs, uint8_t* plane, uint32_t stride, int16_t QFS0, uint16_t W[64], uint8_t quantizer_scale, int intra_dc_precision) {
    __m128i buffer[8];
    for (int i = 0; i < 8; i++) {
        //buffer[i] = _mm_load_si128((__m128i*) & QF[i * 8]);
        buffer[i] = _mm_setzero_si128();
    }
    parse_block<use_dct_one_table, intra>(buffer, m_bs);

    // inverse alt scan and dequant
    inverse_alt_scan_dequant_template_sse2<intra>(buffer, W, quantizer_scale);

    if (intra)
    {
        int32_t res = QFS0 << (3 - intra_dc_precision);
        buffer[0] = _mm_insert_epi16(buffer[0], res, 0);
    }

    // idct
    idct_1d_sse2(buffer);
    transpose_8x8_sse2(buffer);
    idct_1d_sse2(buffer);

    for (int i = 0; i < 4; i++) {
        __m128i tmp, b0, b1;
        b0 = _mm_srai_epi16(buffer[i * 2], 6);
        b1 = _mm_srai_epi16(buffer[i * 2 + 1], 6);
        if (add) {
            __m128i dstl = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*) & plane[(i * 2 + 0) * stride]), _mm_setzero_si128());
            __m128i dsth = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*) & plane[(i * 2 + 1) * stride]), _mm_setzero_si128());
            tmp = _mm_packus_epi16(_mm_adds_epi16(dstl, b0), _mm_adds_epi16(dsth, b1));
        }
        else
            tmp = _mm_packus_epi16(b0, b1);

        _mm_storel_epi64((__m128i*) & plane[(i * 2 + 0) * stride], tmp);
        _mm_storel_epi64((__m128i*) & plane[(i * 2 + 1) * stride], _mm_srli_si128(tmp, 8));
    }
}