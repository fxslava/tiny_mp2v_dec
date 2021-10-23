// Copyright � 2021 Vladislav Ovchinnikov. All rights reserved.
#pragma once
#include <fstream>
#include <vector>
#include "core/common/cpu.hpp"

#if defined(CPU_PLATFORM_X64)
#include <emmintrin.h>
#elif defined(CPU_PLATFORM_AARCH64)
#include "arm_neon.h"
#endif

#define BITSTREAM(bs) \
uint32_t*& bit_ptr = bs->get_ptr(); \
uint64_t & bit_buf = bs->get_buf(); \
uint32_t & bit_idx = bs->get_idx(); \

#define GET_NEXT_BITS(len) ((bit_buf << bit_idx) >> (64 - len))
#define SKIP_BITS(len) bit_idx += len
#define UPDATE_BITS() if (bit_idx >= 32) { bit_buf <<= 32; bit_buf |= (uint64_t)bswap_32(*(bit_ptr++)); bit_idx -= 32; }

class bitstream_reader_c {
private:
    uint32_t* buffer_ptr = nullptr;
    uint64_t  buffer = 0;
    uint32_t  buffer_idx = 64;

    MP2V_INLINE void update_buffer() {
        if (buffer_idx >= 32) {
            buffer <<= 32;
            buffer |= (uint64_t)bswap_32(*(buffer_ptr++));
            buffer_idx -= 32;
        };
    }
public:
    bitstream_reader_c() {}
    ~bitstream_reader_c() {}

    void set_bitstream_buffer(uint8_t* bitstream_buffer) { buffer_ptr = (uint32_t*)bitstream_buffer; }

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

    MP2V_INLINE void skip_bits(int len) {
        buffer_idx += len;
    }

    uint32_t*& get_ptr() { return buffer_ptr; }
    uint64_t & get_buf() { return buffer; }
    uint32_t & get_idx() { return buffer_idx; }
};
