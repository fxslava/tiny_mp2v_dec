// Copyright � 2021 Vladislav Ovchinnikov. All rights reserved.
#pragma once
#include <fstream>
#include <vector>
#include "core/common/cpu.hpp"

#define BITSTREAM(bs) \
uint32_t*& bit_ptr = bs->get_ptr(); \
uint64_t & bit_buf = bs->get_buf(); \
uint32_t & bit_idx = bs->get_idx(); \

#define GET_NEXT_BITS(len) ((bit_buf << bit_idx) >> (64 - len))
#define SKIP_BITS(len) bit_idx += len
#define UPDATE_BITS() if (bit_idx >= 32) { bit_buf <<= 32; bit_buf |= (uint64_t)bswap_32(*(bit_ptr++)); bit_idx -= 32; }

class bitstream_reader_c {
private:
    MP2V_INLINE void read32() {
        buffer <<= 32;
        buffer |= (uint64_t)bswap_32(*(buffer_ptr++));
        buffer_idx -= 32;
    }

    MP2V_INLINE void update_buffer() {
        if (buffer_idx >= 32)
            read32();
    }
public:
    bitstream_reader_c(std::string filename) :
        buffer(0),
        buffer_idx(64)
    {
        std::ifstream fp(filename, std::ios::binary);

        // Calculate size of buffer
        fp.seekg(0, std::ios_base::end);
        std::size_t size = fp.tellg();
        size = ((size + 15) & (~15)) + 2;
        fp.seekg(0, std::ios_base::beg);

        // Allocate buffer
        buffer_pool.resize(size / sizeof(uint32_t));
        buffer_ptr = &buffer_pool[0];
        buffer_end = &buffer_pool[buffer_pool.size() - 1];

        // read file
        fp.read((char*)buffer_ptr, size);
        fp.close();

        generate_start_codes_tbl();
    }

    ~bitstream_reader_c() {}

    MP2V_INLINE uint32_t get_next_bits(int len) {
        update_buffer();
        uint64_t tmp = buffer << buffer_idx;
        return tmp >> (64 - len);
    }

    MP2V_INLINE uint32_t read_next_bits(int len) {
        uint32_t tmp = get_next_bits(len);
        buffer_idx += len;
        return tmp;
    }

    MP2V_INLINE uint32_t get_next_start_code() {
        buffer_idx = 32;
        buffer_ptr = start_code_tbl[start_code_idx] + 1;
        buffer = (uint64_t)bswap_32(*start_code_tbl[start_code_idx++]);
        return buffer;
    }

    MP2V_INLINE void skip_bits(int len) {
        buffer_idx += len;
    }

    uint32_t*& get_ptr() { return buffer_ptr; }
    uint64_t & get_buf() { return buffer; }
    uint32_t & get_idx() { return buffer_idx; }
private:
#if defined(CPU_PLATFORM_X64)
#include <emmintrin.h>
    void generate_start_codes_tbl() {
        static const __m128i pattern_0 = _mm_setzero_si128();
        static const __m128i pattern_1 = _mm_set1_epi8(1);

        for (int i = 0; i < buffer_pool.size(); i += 4) {
            uint8_t* ptr = (uint8_t*)&buffer_pool[i];

            __m128i tmp0 = _mm_cmpeq_epi8(_mm_loadu_si128((__m128i*)(ptr + 0)), pattern_0);
            __m128i tmp1 = _mm_cmpeq_epi8(_mm_loadu_si128((__m128i*)(ptr + 1)), pattern_0);
            __m128i tmp2 = _mm_cmpeq_epi8(_mm_loadu_si128((__m128i*)(ptr + 2)), pattern_1);
            int mask = _mm_movemask_epi8(_mm_and_si128(_mm_and_si128(tmp0, tmp1), tmp2));

            while (mask) {
                int zcnt = bit_scan_forward(mask);
                mask >>= (zcnt + 1);
                ptr += zcnt;
                start_code_tbl.push_back((uint32_t*)ptr++);
            }
        }
    }
#elif defined(CPU_PLATFORM_AARCH64)
#include "arm_neon.h"
    int vmovmaskq_u8(uint8x16_t input)
    {
        uint16x8_t high_bits = vreinterpretq_u16_u8(vshrq_n_u8(input, 7));
        uint32x4_t paired16 = vreinterpretq_u32_u16(vsraq_n_u16(high_bits, high_bits, 7));
        uint64x2_t paired32 = vreinterpretq_u64_u32(vsraq_n_u32(paired16, paired16, 14));
        uint8x16_t paired64 = vreinterpretq_u8_u64 (vsraq_n_u64(paired32, paired32, 28));
        return vgetq_lane_u8(paired64, 0) | ((int)vgetq_lane_u8(paired64, 8) << 8);
    }

    void generate_start_codes_tbl() {
        static const uint8x16_t pattern_0 = vdupq_n_u8(0);
        static const uint8x16_t pattern_1 = vdupq_n_u8(1);
        static const uint8x8_t  half = vdup_n_u8(0x0f);

        for (int i = 0; i < buffer_pool.size(); i += 4) {
            uint8_t* ptr = (uint8_t*)&buffer_pool[i];

            const uint8x16_t tmp0 = vceqq_u8(vld1q_u8(ptr + 0), pattern_0);
            const uint8x16_t tmp1 = vceqq_u8(vld1q_u8(ptr + 1), pattern_0);
            const uint8x16_t tmp2 = vceqq_u8(vld1q_u8(ptr + 2), pattern_1);
            int mask = vmovmaskq_u8(vandq_u8(vandq_u8(tmp0, tmp1), tmp2));

            while (mask) {
                int zcnt = bit_scan_forward(mask);
                mask >>= (zcnt + 1);
                ptr += zcnt;
                start_code_tbl.push_back((uint32_t*)ptr++);
            }
        }
    }
#else
    void generate_start_codes_tbl() {
        auto buf_start = (uint8_t*)buffer_ptr;
        auto buf_end = (uint8_t*)buffer_end;
        int zcnt = 0;
        for (auto ptr = buf_start; ptr < buf_end; ptr++) {
            if (*ptr == 0) zcnt++;
            else {
                if ((*ptr == 1) && (zcnt >= 2))
                    start_code_tbl.push_back((uint32_t*)(ptr - 2));
                zcnt = 0;
            }
        }
    }
#endif

private:
    //FILE* bitstream = nullptr;
    std::vector<uint32_t*> start_code_tbl;
    std::vector<uint32_t, AlignmentAllocator<uint8_t, 32>> buffer_pool;
    uint32_t* buffer_ptr = nullptr;
    uint32_t* buffer_end = nullptr;
    uint64_t  buffer = 0;
    uint32_t  buffer_idx = 64;
    uint32_t  start_code_idx = 0;
};