// Copyright � 2021 Vladislav Ovchinnikov. All rights reserved.
#pragma once
#include <stdint.h>
#include "arm_neon.h"
#include "common/cpu.hpp"

MP2V_INLINE int16x8_t vmul_coeff_s16(int16x8_t src, int32_t coeff) {
    const int32x4_t vcof = vdupq_n_s32(coeff);
    const int32x4_t src0 = vmovl_s16(vget_low_s16(src));
    const int32x4_t src1 = vmovl_s16(vget_high_s16(src));
    const int16x4_t res0 = vshrn_n_s32(vmulq_s32(src0, vcof), 16);
    const int16x4_t res1 = vshrn_n_s32(vmulq_s32(src1, vcof), 16);
    return vcombine_s16(res0, res1);
}

MP2V_INLINE void idct_1d_aarch64(int16x8_t(&src)[8]) {
    // step 0
    const int16x8_t v15 = vmul_coeff_s16(src[0], 185364);
    const int16x8_t v26 = vmul_coeff_s16(src[1], 257107);
    const int16x8_t v21 = vmul_coeff_s16(src[2], 242189);
    const int16x8_t v28 = vmul_coeff_s16(src[3], 217965);
    const int16x8_t v16 = vmul_coeff_s16(src[4], 185364);
    const int16x8_t v25 = vmul_coeff_s16(src[5], 145639);
    const int16x8_t v22 = vmul_coeff_s16(src[6], 100318);
    const int16x8_t v27 = vmul_coeff_s16(src[7],  51142);
    // step 1
    const int16x8_t v19 = vsubq_s16(v25, v28); // /2
    const int16x8_t v20 = vsubq_s16(v26, v27); // /2
    const int16x8_t v23 = vaddq_s16(v26, v27); // /2
    const int16x8_t v24 = vaddq_s16(v25, v28); // /2
    const int16x8_t v7  = vaddq_s16(v23, v24); // /4
    const int16x8_t v11 = vaddq_s16(v21, v22); // /2
    const int16x8_t v13 = vsubq_s16(v23, v24); // /4
    const int16x8_t v17 = vsubq_s16(v21, v22); // /2
    const int16x8_t v8  = vaddq_s16(v15, v16); // /2
    const int16x8_t v9  = vsubq_s16(v15, v16); // /2
    // step 2
    const int16x8_t v18 = vmul_coeff_s16(vsubq_s16(v19, v20), 25079); //(v19 - v20) * s1[4]; /2
    const int16x8_t v12 = vsubq_s16(v18, vmul_coeff_s16(v19, 85626)); // v18 - v19 * s1[3];  /2
    const int16x8_t v14 = vsubq_s16(vmul_coeff_s16(v20, 35468), v18); // v20 * s1[1] - v18); /2
    const int16x8_t v6  = vsubq_s16(vshlq_n_s16(v14, 1), v7);         // v14 - v7            /4
    const int16x8_t v5  = vsubq_s16(vmul_coeff_s16(v13, 92681), v6);  // v13 / s1[2] - v6;   /4
    const int16x8_t v4  = vaddq_s16(v5, vshlq_n_s16(v12, 1));         // v5 + v12;           /4
    const int16x8_t v10 = vsubq_s16(vmul_coeff_s16(v17, 92681), v11); // v17 / s1[0] - v11;  /2
    const int16x8_t v0  = vaddq_s16(v8, v11); // /4
    const int16x8_t v1  = vaddq_s16(v9, v10); // /4
    const int16x8_t v2  = vsubq_s16(v9, v10); // /4
    const int16x8_t v3  = vsubq_s16(v8, v11); // /4
    // step 3
    src[0] = vaddq_s16(v0, v7); // /8
    src[1] = vaddq_s16(v1, v6); // /8
    src[2] = vaddq_s16(v2, v5); // /8
    src[3] = vsubq_s16(v3, v4); // /8
    src[4] = vaddq_s16(v3, v4); // /8
    src[5] = vsubq_s16(v2, v5); // /8
    src[6] = vsubq_s16(v1, v6); // /8
    src[7] = vsubq_s16(v0, v7); // /8
}

MP2V_INLINE void transpose_8x8_aarch64(int16x8_t(&src)[8]) {
    int16x8_t V8  = vtrn1q_s16(src[0], src[1]);
    int16x8_t V9  = vtrn2q_s16(src[0], src[1]);
    int16x8_t V10 = vtrn1q_s16(src[2], src[3]);
    int16x8_t V11 = vtrn2q_s16(src[2], src[3]);
    int16x8_t V12 = vtrn1q_s16(src[4], src[5]);
    int16x8_t V13 = vtrn2q_s16(src[4], src[5]);
    int16x8_t V14 = vtrn1q_s16(src[6], src[7]);
    int16x8_t V15 = vtrn2q_s16(src[6], src[7]);
    uint32x4_t V0  = vtrn1q_u32(vreinterpretq_u32_s16(V8),  vreinterpretq_u32_s16(V10));
    uint32x4_t V1  = vtrn1q_u32(vreinterpretq_u32_s16(V9),  vreinterpretq_u32_s16(V11));
    uint32x4_t V2  = vtrn2q_u32(vreinterpretq_u32_s16(V8),  vreinterpretq_u32_s16(V10));
    uint32x4_t V3  = vtrn2q_u32(vreinterpretq_u32_s16(V9),  vreinterpretq_u32_s16(V11));
    uint32x4_t V4  = vtrn1q_u32(vreinterpretq_u32_s16(V12), vreinterpretq_u32_s16(V14));
    uint32x4_t V5  = vtrn1q_u32(vreinterpretq_u32_s16(V13), vreinterpretq_u32_s16(V15));
    uint32x4_t V6  = vtrn2q_u32(vreinterpretq_u32_s16(V12), vreinterpretq_u32_s16(V14));
    uint32x4_t V7  = vtrn2q_u32(vreinterpretq_u32_s16(V13), vreinterpretq_u32_s16(V15));
    src[0] = vreinterpretq_s16_u64(vtrn1q_u64(vreinterpretq_u64_u32(V0), vreinterpretq_u64_u32(V4)));
    src[1] = vreinterpretq_s16_u64(vtrn1q_u64(vreinterpretq_u64_u32(V1), vreinterpretq_u64_u32(V5)));
    src[2] = vreinterpretq_s16_u64(vtrn1q_u64(vreinterpretq_u64_u32(V2), vreinterpretq_u64_u32(V6)));
    src[3] = vreinterpretq_s16_u64(vtrn1q_u64(vreinterpretq_u64_u32(V3), vreinterpretq_u64_u32(V7)));
    src[4] = vreinterpretq_s16_u64(vtrn2q_u64(vreinterpretq_u64_u32(V0), vreinterpretq_u64_u32(V4)));
    src[5] = vreinterpretq_s16_u64(vtrn2q_u64(vreinterpretq_u64_u32(V1), vreinterpretq_u64_u32(V5)));
    src[6] = vreinterpretq_s16_u64(vtrn2q_u64(vreinterpretq_u64_u32(V2), vreinterpretq_u64_u32(V6)));
    src[7] = vreinterpretq_s16_u64(vtrn2q_u64(vreinterpretq_u64_u32(V3), vreinterpretq_u64_u32(V7)));
}

template<bool add>
void inverse_dct_template(uint8_t* plane, int16_t F[64], int stride) {
    int16x8_t buffer[8];
    for (int i = 0; i < 8; i++)
        buffer[i] = vld1q_s16(&F[i*8]);

    idct_1d_aarch64(buffer);
    transpose_8x8_aarch64(buffer);
    idct_1d_aarch64(buffer);

    for (int i = 0; i < 4; i++) {
        if (add) {
            int16x8_t b0 = vshrq_n_s16(buffer[i * 2], 6);
            int16x8_t b1 = vshrq_n_s16(buffer[i * 2 + 1], 6);
            b0 = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(b0), vld1_u8(&plane[(i * 2 + 0) * stride])));
            b1 = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(b1), vld1_u8(&plane[(i * 2 + 1) * stride])));
            vst1_u8(&plane[(i * 2 + 0) * stride], vqmovun_s16(b0));
            vst1_u8(&plane[(i * 2 + 1) * stride], vqmovun_s16(b1));
        }
        else {
            vst1_u8(&plane[(i * 2 + 0) * stride], vqshrun_n_s16(buffer[i * 2], 6));
            vst1_u8(&plane[(i * 2 + 1) * stride], vqshrun_n_s16(buffer[i * 2 + 1], 6));
        }
    }
}
