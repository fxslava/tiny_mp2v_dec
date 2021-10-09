// Copyright � 2021 Vladislav Ovchinnikov. All rights reserved.
#include <vector>
#include <random>
#include <algorithm>
#include <iterator>
#include <chrono>

// unit test common
#include "test_common.h"

// Tiny MPEG2 MC headers
#include "core/idct_ref.hpp"
#if defined(CPU_PLATFORM_X64)
#include "core/idct_sse2.hpp"
#elif defined(CPU_PLATFORM_AARCH64)
#include "core/idct_aarch64.hpp"
#endif

constexpr int TEST_NUM_ITERATIONS = 10;
constexpr int TEST_NUM_ITERATIONS_PERFORMANCE = 10000;
constexpr int IDCT_PLANE_SIZE = 8;
constexpr int IDCT_PLANE_STRIDE = 32;
constexpr int IDCT_PIXEL_MIN_VALUE = -255;
constexpr int IDCT_PIXEL_MAX_VALUE = 255;
constexpr int IDCT_RANDOM_SEED = 1729;

typedef void (*idct_func_t)(uint8_t* plane, int16_t F[64], int stride);

class simd_idct_test_c : public ::testing::Test {
public:
    simd_idct_test_c() : gen(IDCT_RANDOM_SEED) {
        dst_plane.resize(IDCT_PLANE_SIZE * IDCT_PLANE_STRIDE);
        dst_plane_ref.resize(IDCT_PLANE_SIZE * IDCT_PLANE_STRIDE);
    }
    ~simd_idct_test_c() {}
    void SetUp() {
        std::fill(dst_plane.begin(), dst_plane.end(), 0);
        std::fill(dst_plane_ref.begin(), dst_plane_ref.end(), 0);
    }
    void TearDown() {}

    bool test_idct(idct_func_t func_ref, idct_func_t func_simd) {
        for (int step = 0; step < TEST_NUM_ITERATIONS; step++) {
            generate_sources();
            func_ref (&dst_plane_ref[0], src_plane, IDCT_PLANE_STRIDE);
            func_simd(&dst_plane[0],     src_plane, IDCT_PLANE_STRIDE);
            if (dst_plane != dst_plane_ref)
                return false;
        }
        return true;
    }

    void generate_sources() {
        std::uniform_int_distribution<int16_t> uniform_gen(0, IDCT_PIXEL_MAX_VALUE);
        for (auto& val : src_plane)
            val = uniform_gen(gen);
    }

protected:
    ALIGN(32) int16_t src_plane[64]; // unaligned
    std::vector<uint8_t, AlignmentAllocator<uint8_t, 32>> dst_plane;
    std::vector<uint8_t, AlignmentAllocator<uint8_t, 32>> dst_plane_ref;
    std::mt19937 gen{};
};

#define TEST_IDCT_ROUTINES(simd) \
TEST_F(simd_idct_test_c, validation_idct_add_##simd) { EXPECT_TRUE(test_idct(inverse_dct_template_ref<true>,  inverse_dct_template<true> )); } \
TEST_F(simd_idct_test_c, validation_idct_mov_##simd) { EXPECT_TRUE(test_idct(inverse_dct_template_ref<false>, inverse_dct_template<false>)); }

#if defined(CPU_PLATFORM_X64)
TEST_IDCT_ROUTINES(sse2);
#elif defined(CPU_PLATFORM_AARCH64)
TEST_IDCT_ROUTINES(aarch64);
#endif
