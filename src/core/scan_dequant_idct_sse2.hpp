// Copyright © 2021 Vladislav Ovchinnikov. All rights reserved.
#pragma once
#include <stdint.h>
#include <emmintrin.h>
#include "common/cpu.hpp"
#include "scan_sse4.hpp"
#include "idct_sse2.hpp"

template<bool intra, bool add>
void scan_dequant_idct_template_sse2(uint8_t* plane, uint32_t stride, int16_t QF0, int8_t QF[64], uint16_t W[64], uint8_t quantizer_scale, int intra_dc_precision) {
    ALIGN(32) __m128i buffer[8];
    
    for (int i = 0; i < 8; i++)
        buffer[i] = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*) & QF[i * 8]));

    // inverse alt scan and dequant
    inverse_alt_scan_dequant_template_sse2<intra>(buffer, W, quantizer_scale);

    if (intra)
    {
        int32_t res = QF0 << (3 - intra_dc_precision);
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