#include "mc.h"
#include <emmintrin.h>
#include "common/cpu.hpp"

template<mc_type_e mc_type>
MP2V_INLINE __m128i mc8_func_template_sse2(uint8_t* src, uint32_t stride) {
    switch (mc_type)
    {
    case MC_00:
        return _mm_loadl_epi64((__m128i*)src);
    case MC_01:
        return _mm_avg_epu8(_mm_loadl_epi64((__m128i*)src), _mm_loadl_epi64((__m128i*) & src[1]));
    case MC_10:
        return _mm_avg_epu8(_mm_loadl_epi64((__m128i*)src), _mm_loadl_epi64((__m128i*) & src[stride]));
    case MC_11:
    default:
        __m128i tmp0 = _mm_avg_epu8(_mm_loadl_epi64((__m128i*)src), _mm_loadl_epi64((__m128i*) & src[1]));
        __m128i tmp1 = _mm_avg_epu8(_mm_loadl_epi64((__m128i*) & src[stride]), _mm_loadl_epi64((__m128i*) & src[stride + 1]));
        return _mm_avg_epu8(tmp0, tmp1);
    }
}

template<mc_type_e mc_type>
MP2V_INLINE __m128i mc16_func_template_sse2(uint8_t* src, uint32_t stride) {
    switch (mc_type)
    {
    case MC_00:
        return _mm_loadu_si128((__m128i*)src);
    case MC_01:
        return _mm_avg_epu8(_mm_loadu_si128((__m128i*)src), _mm_loadu_si128((__m128i*) & src[1]));
    case MC_10:
        return _mm_avg_epu8(_mm_loadu_si128((__m128i*)src), _mm_loadu_si128((__m128i*) & src[stride]));
    case MC_11:
    default:
        __m128i tmp0 = _mm_avg_epu8(_mm_loadu_si128((__m128i*)src), _mm_loadu_si128((__m128i*) & src[1]));
        __m128i tmp1 = _mm_avg_epu8(_mm_loadu_si128((__m128i*) & src[stride]), _mm_loadu_si128((__m128i*) & src[stride + 1]));
        return _mm_avg_epu8(tmp0, tmp1);
    }
}

template<mc_type_e mc_type>
MP2V_INLINE void pred_mc16_template_sse2(uint8_t* dst, uint8_t* src, uint32_t stride, int height)
{
    for (int j = 0; j < height; j += 8) {
        _mm_store_si128((__m128i*)(dst + stride * 0), mc16_func_template_sse2<mc_type>(src + stride * 0, stride));
        _mm_store_si128((__m128i*)(dst + stride * 1), mc16_func_template_sse2<mc_type>(src + stride * 1, stride));
        _mm_store_si128((__m128i*)(dst + stride * 2), mc16_func_template_sse2<mc_type>(src + stride * 2, stride));
        _mm_store_si128((__m128i*)(dst + stride * 3), mc16_func_template_sse2<mc_type>(src + stride * 3, stride));
        _mm_store_si128((__m128i*)(dst + stride * 4), mc16_func_template_sse2<mc_type>(src + stride * 4, stride));
        _mm_store_si128((__m128i*)(dst + stride * 5), mc16_func_template_sse2<mc_type>(src + stride * 5, stride));
        _mm_store_si128((__m128i*)(dst + stride * 6), mc16_func_template_sse2<mc_type>(src + stride * 6, stride));
        _mm_store_si128((__m128i*)(dst + stride * 7), mc16_func_template_sse2<mc_type>(src + stride * 7, stride));
        src += stride * 8; dst += stride * 8;
    }
}

template<mc_type_e mc_type>
MP2V_INLINE void pred_mc8_template_sse2(uint8_t* dst, uint8_t* src, uint32_t stride, int height)
{
    for (int j = 0; j < height; j += 4) {
        _mm_storel_epi64((__m128i*)(dst + stride * 0), mc8_func_template_sse2<mc_type>(src + stride * 0, stride));
        _mm_storel_epi64((__m128i*)(dst + stride * 1), mc8_func_template_sse2<mc_type>(src + stride * 1, stride));
        _mm_storel_epi64((__m128i*)(dst + stride * 2), mc8_func_template_sse2<mc_type>(src + stride * 2, stride));
        _mm_storel_epi64((__m128i*)(dst + stride * 3), mc8_func_template_sse2<mc_type>(src + stride * 3, stride));
        src += stride * 4; dst += stride * 4;
    }
}

void mc_pred00_16xh_sse2(uint8_t* dst, uint8_t* src, uint32_t stride, uint32_t height) { pred_mc16_template_sse2<MC_00>(dst, src, stride, height); }
void mc_pred01_16xh_sse2(uint8_t* dst, uint8_t* src, uint32_t stride, uint32_t height) { pred_mc16_template_sse2<MC_01>(dst, src, stride, height); }
void mc_pred10_16xh_sse2(uint8_t* dst, uint8_t* src, uint32_t stride, uint32_t height) { pred_mc16_template_sse2<MC_10>(dst, src, stride, height); }
void mc_pred11_16xh_sse2(uint8_t* dst, uint8_t* src, uint32_t stride, uint32_t height) { pred_mc16_template_sse2<MC_11>(dst, src, stride, height); }
void mc_pred00_8xh_sse2 (uint8_t* dst, uint8_t* src, uint32_t stride, uint32_t height) { pred_mc8_template_sse2<MC_00>(dst, src, stride, height); }
void mc_pred01_8xh_sse2 (uint8_t* dst, uint8_t* src, uint32_t stride, uint32_t height) { pred_mc8_template_sse2<MC_01>(dst, src, stride, height); }
void mc_pred10_8xh_sse2 (uint8_t* dst, uint8_t* src, uint32_t stride, uint32_t height) { pred_mc8_template_sse2<MC_10>(dst, src, stride, height); }
void mc_pred11_8xh_sse2 (uint8_t* dst, uint8_t* src, uint32_t stride, uint32_t height) { pred_mc8_template_sse2<MC_11>(dst, src, stride, height); }

template<mc_type_e mc_type_src0, mc_type_e mc_type_src1>
MP2V_INLINE void bidir_mc16_line_template_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride) {
    __m128i tmp0 = mc16_func_template_sse2<mc_type_src0>(src0, stride);
    __m128i tmp1 = mc16_func_template_sse2<mc_type_src1>(src1, stride);
    __m128i res = _mm_avg_epu8(tmp0, tmp1);
    _mm_store_si128((__m128i*)dst, res);
}

template<mc_type_e mc_type_src0, mc_type_e mc_type_src1>
MP2V_INLINE void bidir_mc16_template_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, int height)
{
    for (int j = 0; j < height; j += 8) {
        bidir_mc16_line_template_sse2<mc_type_src0, mc_type_src1>(dst + stride * 0, src0 + stride * 0, src1 + stride * 0, stride);
        bidir_mc16_line_template_sse2<mc_type_src0, mc_type_src1>(dst + stride * 1, src0 + stride * 1, src1 + stride * 1, stride);
        bidir_mc16_line_template_sse2<mc_type_src0, mc_type_src1>(dst + stride * 2, src0 + stride * 2, src1 + stride * 2, stride);
        bidir_mc16_line_template_sse2<mc_type_src0, mc_type_src1>(dst + stride * 3, src0 + stride * 3, src1 + stride * 3, stride);
        bidir_mc16_line_template_sse2<mc_type_src0, mc_type_src1>(dst + stride * 4, src0 + stride * 4, src1 + stride * 4, stride);
        bidir_mc16_line_template_sse2<mc_type_src0, mc_type_src1>(dst + stride * 5, src0 + stride * 5, src1 + stride * 5, stride);
        bidir_mc16_line_template_sse2<mc_type_src0, mc_type_src1>(dst + stride * 6, src0 + stride * 6, src1 + stride * 6, stride);
        bidir_mc16_line_template_sse2<mc_type_src0, mc_type_src1>(dst + stride * 7, src0 + stride * 7, src1 + stride * 7, stride);
        dst += stride * 8; src0 += stride * 8; src1 += stride * 8;
    }
}

template<mc_type_e mc_type_src0, mc_type_e mc_type_src1>
MP2V_INLINE void bidir_mc8_line_template_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride) {
    __m128i tmp0 = mc8_func_template_sse2<mc_type_src0>(src0, stride);
    __m128i tmp1 = mc8_func_template_sse2<mc_type_src1>(src1, stride);
    __m128i res = _mm_avg_epu8(tmp0, tmp1);
    _mm_storel_epi64((__m128i*)dst, res);
}

template<mc_type_e mc_type_src0, mc_type_e mc_type_src1>
MP2V_INLINE void bidir_mc8_template_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, int height)
{
    for (int j = 0; j < height; j += 4) {
        bidir_mc8_line_template_sse2<mc_type_src0, mc_type_src1>(dst + stride * 0, src0 + stride * 0, src1 + stride * 0, stride);
        bidir_mc8_line_template_sse2<mc_type_src0, mc_type_src1>(dst + stride * 1, src0 + stride * 1, src1 + stride * 1, stride);
        bidir_mc8_line_template_sse2<mc_type_src0, mc_type_src1>(dst + stride * 2, src0 + stride * 2, src1 + stride * 2, stride);
        bidir_mc8_line_template_sse2<mc_type_src0, mc_type_src1>(dst + stride * 3, src0 + stride * 3, src1 + stride * 3, stride);
        dst += stride * 4; src0 += stride * 4; src1 += stride * 4;
    }
}

void mc_bidir0000_16xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc16_template_sse2<MC_00, MC_00>(dst, src0, src1, stride, height); }
void mc_bidir0001_16xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc16_template_sse2<MC_00, MC_01>(dst, src0, src1, stride, height); }
void mc_bidir0010_16xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc16_template_sse2<MC_00, MC_10>(dst, src0, src1, stride, height); }
void mc_bidir0011_16xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc16_template_sse2<MC_00, MC_11>(dst, src0, src1, stride, height); }
void mc_bidir0100_16xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc16_template_sse2<MC_01, MC_00>(dst, src0, src1, stride, height); }
void mc_bidir0101_16xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc16_template_sse2<MC_01, MC_01>(dst, src0, src1, stride, height); }
void mc_bidir0110_16xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc16_template_sse2<MC_01, MC_10>(dst, src0, src1, stride, height); }
void mc_bidir0111_16xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc16_template_sse2<MC_01, MC_11>(dst, src0, src1, stride, height); }
void mc_bidir1000_16xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc16_template_sse2<MC_10, MC_00>(dst, src0, src1, stride, height); }
void mc_bidir1001_16xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc16_template_sse2<MC_10, MC_01>(dst, src0, src1, stride, height); }
void mc_bidir1010_16xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc16_template_sse2<MC_10, MC_10>(dst, src0, src1, stride, height); }
void mc_bidir1011_16xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc16_template_sse2<MC_10, MC_11>(dst, src0, src1, stride, height); }
void mc_bidir1100_16xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc16_template_sse2<MC_11, MC_00>(dst, src0, src1, stride, height); }
void mc_bidir1101_16xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc16_template_sse2<MC_11, MC_01>(dst, src0, src1, stride, height); }
void mc_bidir1110_16xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc16_template_sse2<MC_11, MC_10>(dst, src0, src1, stride, height); }
void mc_bidir1111_16xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc16_template_sse2<MC_11, MC_11>(dst, src0, src1, stride, height); }
void mc_bidir0000_8xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc8_template_sse2<MC_00, MC_00>(dst, src0, src1, stride, height); }
void mc_bidir0001_8xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc8_template_sse2<MC_00, MC_01>(dst, src0, src1, stride, height); }
void mc_bidir0010_8xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc8_template_sse2<MC_00, MC_10>(dst, src0, src1, stride, height); }
void mc_bidir0011_8xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc8_template_sse2<MC_00, MC_11>(dst, src0, src1, stride, height); }
void mc_bidir0100_8xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc8_template_sse2<MC_01, MC_00>(dst, src0, src1, stride, height); }
void mc_bidir0101_8xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc8_template_sse2<MC_01, MC_01>(dst, src0, src1, stride, height); }
void mc_bidir0110_8xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc8_template_sse2<MC_01, MC_10>(dst, src0, src1, stride, height); }
void mc_bidir0111_8xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc8_template_sse2<MC_01, MC_11>(dst, src0, src1, stride, height); }
void mc_bidir1000_8xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc8_template_sse2<MC_10, MC_00>(dst, src0, src1, stride, height); }
void mc_bidir1001_8xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc8_template_sse2<MC_10, MC_01>(dst, src0, src1, stride, height); }
void mc_bidir1010_8xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc8_template_sse2<MC_10, MC_10>(dst, src0, src1, stride, height); }
void mc_bidir1011_8xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc8_template_sse2<MC_10, MC_11>(dst, src0, src1, stride, height); }
void mc_bidir1100_8xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc8_template_sse2<MC_11, MC_00>(dst, src0, src1, stride, height); }
void mc_bidir1101_8xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc8_template_sse2<MC_11, MC_01>(dst, src0, src1, stride, height); }
void mc_bidir1110_8xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc8_template_sse2<MC_11, MC_10>(dst, src0, src1, stride, height); }
void mc_bidir1111_8xh_sse2(uint8_t* dst, uint8_t* src0, uint8_t* src1, uint32_t stride, uint32_t height) { bidir_mc8_template_sse2<MC_11, MC_11>(dst, src0, src1, stride, height); }
