#pragma once
#include <vector>
#include <functional>
#include "core/common/cpu.hpp"

#if defined(CPU_PLATFORM_X64)
void scan_start_codes(uint8_t* buffer_ptr, uint8_t* buffer_end, std::function<void(uint8_t*)> func) {
    static const __m128i pattern_0 = _mm_setzero_si128();
    static const __m128i pattern_1 = _mm_set1_epi8(1);

    for (auto dword_ptr = buffer_ptr; dword_ptr < buffer_end; dword_ptr += 16) {
        uint8_t* ptr = dword_ptr;

        __m128i tmp0 = _mm_cmpeq_epi8(_mm_loadu_si128((__m128i*)(ptr + 0)), pattern_0);
        __m128i tmp1 = _mm_cmpeq_epi8(_mm_loadu_si128((__m128i*)(ptr + 1)), pattern_0);
        __m128i tmp2 = _mm_cmpeq_epi8(_mm_loadu_si128((__m128i*)(ptr + 2)), pattern_1);
        int mask = _mm_movemask_epi8(_mm_and_si128(_mm_and_si128(tmp0, tmp1), tmp2));

        while (mask) {
            int zcnt = bit_scan_forward(mask);
            mask >>= (zcnt + 1);
            ptr += zcnt;
            func(ptr++);
        }
    }
}
#else
void scan_start_codes(uint8_t* buffer_ptr, uint8_t* buffer_end, std::function<void(uint8_t*)> func) {
    int zcnt = 0;
    for (auto ptr = buffer_ptr; ptr < buffer_end; ptr++) {
        if (*ptr == 0) zcnt++;
        else {
            if ((*ptr == 1) && (zcnt >= 2))
                func(ptr - 2);
            zcnt = 0;
        }
    }
}
#endif