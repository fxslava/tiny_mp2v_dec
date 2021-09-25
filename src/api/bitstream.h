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
        uint32_t tmp = *buffer_ptr;
        buffer_ptr++;
        buffer |= (uint64_t)bswap_32(tmp);
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
    std::vector<uint32_t, AlignmentAllocator<uint8_t, 32>> buffer_pool;
    ALIGN(16) uint32_t* buffer_ptr = nullptr;
    ALIGN(16) uint32_t* buffer_end = nullptr;
    ALIGN(16) uint64_t  buffer = 0;
    ALIGN(16) uint32_t  buffer_idx = 64;
};