// Copyright © 2021 Vladislav Ovchinnikov. All rights reserved.
#pragma once
#include <stdint.h>
#include <emmintrin.h>
#include "common/cpu.hpp"
#include <algorithm>

template<bool intra>
MP2V_INLINE __m128i inverse_quant_scalar_sse2(const __m128i QF, const __m128i W_quantizer_scale) {
    __m128i tmp;
    if (!intra) {
        const __m128i k = _mm_and_si128(_mm_sign_epi16(_mm_set1_epi16(1), QF), _mm_cmpeq_epi16(QF, _mm_setzero_si128()));
        tmp = _mm_add_epi16(_mm_slli_epi16(QF, 1), k);
    }
    else
        tmp = _mm_slli_epi16(QF, 1);

    const __m128i resl = _mm_mullo_epi16(tmp, W_quantizer_scale);
    const __m128i resh = _mm_mulhi_epi16(tmp, W_quantizer_scale);
    const __m128i res = _mm_or_si128(_mm_slli_epi16(resh, 11), _mm_srli_epi16(resl, 5));
    return _mm_max_epi16(_mm_min_epi16(res, _mm_set1_epi16(2047)), _mm_set1_epi16(-2048));
}

MP2V_INLINE __m128i _mm_mmctl_epi16(const __m128i src) {
    __m128i tmp;
    tmp = _mm_xor_si128(src, _mm_bslli_si128(src, 1));
    tmp = _mm_xor_si128(tmp, _mm_bslli_si128(tmp, 2));
    tmp = _mm_xor_si128(tmp, _mm_bslli_si128(tmp, 4));
    tmp = _mm_xor_si128(tmp, _mm_bslli_si128(tmp, 8));
    return tmp;
}

template<bool intra, bool alt_scan>
void dequant_sse2(int16_t* QFS, int8_t qfs[64], int8_t qid[64], uint8_t w[64], int n, uint8_t quantizer_scale) {
    ALIGN(16) int16_t tmp[8];
    int i = 0;
    if (n & ~7)
        for (; i < (n & ~7); i += 8, qid += 8) {
            __m128i vqfs = _mm_cvtepi8_epi16(_mm_load_si128((__m128i*) & qfs[i]));
            __m128i vwqt = _mm_unpacklo_epi8(_mm_load_si128((__m128i*) & w[i]), _mm_setzero_si128());
            _mm_store_si128((__m128i*) tmp, inverse_quant_scalar_sse2<intra>(vqfs, _mm_mullo_epi16(vwqt, _mm_set1_epi16(quantizer_scale))));
            for (int j = 0; j < 8; j++) {
                int idx = (int)g_scan_trans[alt_scan ? 1 : 0][qid[j]];
                QFS[idx] = tmp[j];
            }
        }

    if (n & 7) {
        __m128i vqfs = _mm_cvtepi8_epi16(_mm_load_si128((__m128i*) & qfs[i]));
        __m128i vwqt = _mm_unpacklo_epi8(_mm_load_si128((__m128i*) & w[i]), _mm_setzero_si128());
        _mm_store_si128((__m128i*) tmp, inverse_quant_scalar_sse2<intra>(vqfs, _mm_mullo_epi16(vwqt, _mm_set1_epi16(quantizer_scale))));
        for (int j = 0; j < (n & 7); j++) {
            int idx = (int)g_scan_trans[alt_scan ? 1 : 0][qid[j]];
            QFS[idx] = tmp[j];
        }
    }
}