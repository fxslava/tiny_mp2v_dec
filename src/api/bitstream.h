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
    }

    ~bitstream_reader_c() {}

#if (defined(__GNUC__) && defined(__x86_64)) || (defined(_MSC_VER) && defined(_M_X64))
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
#endif

    void generate_start_codes_tbl_c() {
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

    MP2V_INLINE bool seek_pattern(uint32_t pattern, int len) {
        do {
            update_buffer();
            int range = 64 - buffer_idx - len;
            uint64_t tmp = buffer << buffer_idx;
            uint64_t mask = (1ll << len) - 1;
            int offset = 64 - len;
            for (int pos = 0; pos < range; pos++) {
                if ((uint32_t)((tmp >> offset) & mask) == pattern) {
                    buffer_idx += pos;
                    return true;
                }
                tmp <<= 1;
            }
            buffer_idx += range;
        } while (buffer_ptr < buffer_end);
        return false;
    }

    MP2V_INLINE void skip_bits(int len) {
        buffer_idx += len;
    }

    uint32_t*& get_ptr() { return buffer_ptr; }
    uint64_t & get_buf() { return buffer; }
    uint32_t & get_idx() { return buffer_idx; }

private:
    //FILE* bitstream = nullptr;
    std::vector<uint32_t*> start_code_tbl;
    std::vector<uint32_t, AlignmentAllocator<uint8_t, 32>> buffer_pool;
    ALIGN(16) uint32_t* buffer_ptr = nullptr;
    ALIGN(16) uint32_t* buffer_end = nullptr;
    ALIGN(16) uint64_t  buffer = 0;
    ALIGN(16) uint32_t  buffer_idx = 64;
};