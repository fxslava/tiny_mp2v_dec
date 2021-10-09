// Copyright � 2021 Vladislav Ovchinnikov. All rights reserved.
#pragma once
#include <stdint.h>
#include "common/cpu.hpp"

MP2V_INLINE int16_t mul_coeff_s16(int16_t src, int32_t coeff) {
    return ((int32_t)src * coeff) >> 16;
}

MP2V_INLINE void idct_1d_ref(int16_t* dst, int16_t* src) {
    for (int i = 0; i < 8; i++) {
        // step 0
        const int16_t v15 = mul_coeff_s16(src[0 * 8 + i], 185364);
        const int16_t v26 = mul_coeff_s16(src[1 * 8 + i], 257107);
        const int16_t v21 = mul_coeff_s16(src[2 * 8 + i], 242189);
        const int16_t v28 = mul_coeff_s16(src[3 * 8 + i], 217965);
        const int16_t v16 = mul_coeff_s16(src[4 * 8 + i], 185364);
        const int16_t v25 = mul_coeff_s16(src[5 * 8 + i], 145639);
        const int16_t v22 = mul_coeff_s16(src[6 * 8 + i], 100318);
        const int16_t v27 = mul_coeff_s16(src[7 * 8 + i], 51142);
        // step 1
        const int16_t v19 = v25 - v28; // /2
        const int16_t v20 = v26 - v27; // /2
        const int16_t v23 = v26 + v27; // /2
        const int16_t v24 = v25 + v28; // /2
        const int16_t v7 = v23 + v24; // /4
        const int16_t v11 = v21 + v22; // /2
        const int16_t v13 = v23 - v24; // /4
        const int16_t v17 = v21 - v22; // /2
        const int16_t v8 = v15 + v16; // /2
        const int16_t v9 = v15 - v16; // /2
        // step 2
        const int16_t v18 = mul_coeff_s16(v19 - v20, 25079); //(v19 - v20) * s1[4]; /2
        const int16_t v12 = v18 - mul_coeff_s16(v19, 85626); // v18 - v19 * s1[3];  /2
        const int16_t v14 = mul_coeff_s16(v20, 35468) - v18; // v20 * s1[1] - v18); /2
        const int16_t v6 = (v14 << 1) - v7;                 // v14 - v7            /4
        const int16_t v5 = mul_coeff_s16(v13, 92681) - v6;  // v13 / s1[2] - v6;   /4
        const int16_t v4 = v5 + (v12 << 1);                 // v5 + v12;           /4
        const int16_t v10 = mul_coeff_s16(v17, 92681) - v11; // v17 / s1[0] - v11;  /2
        const int16_t v0 = v8 + v11; // /4
        const int16_t v1 = v9 + v10; // /4
        const int16_t v2 = v9 - v10; // /4
        const int16_t v3 = v8 - v11; // /4
        // step 3
        dst[0 * 8 + i] = v0 + v7; // /8
        dst[1 * 8 + i] = v1 + v6; // /8
        dst[2 * 8 + i] = v2 + v5; // /8
        dst[3 * 8 + i] = v3 - v4; // /8
        dst[4 * 8 + i] = v3 + v4; // /8
        dst[5 * 8 + i] = v2 - v5; // /8
        dst[6 * 8 + i] = v1 - v6; // /8
        dst[7 * 8 + i] = v0 - v7; // /8
    }
}

#define SATURATE(val) (((val) < 0) ? 0 : (((val) > 255) ? 255 : (val)))

template<bool add>
void inverse_dct_template_ref(uint8_t* plane, int16_t F[64], int stride) {
    int16_t tmp0[64];
    int16_t tmp1[64];

    idct_1d_ref(tmp0, F);

    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            tmp1[i + j * 8] = tmp0[i * 8 + j];

    idct_1d_ref(tmp0, tmp1);

    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) {
            if (add) plane[i * stride + j] = (uint8_t)SATURATE((int)plane[i * stride + j] + (tmp0[j + i * 8] >> 6));
            else     plane[i * stride + j] = (uint8_t)SATURATE(tmp0[j + i * 8] >> 6);
        }
}
