// Copyright � 2021 Vladislav Ovchinnikov. All rights reserved.
#include <vector>
#include <random>
#include <algorithm>
#include <iterator>
#include <chrono>

using namespace std::chrono;

// unit test common
#include "test_common.h"
#include "threads_test_common.hpp"

#include <windows.h>

constexpr int THREAD_POOL_SIZE = 8;
constexpr int TASK_POOL_SIZE = 8;

class threads_test_c : public ::testing::Test {
public:
    threads_test_c() : queue(TASK_POOL_SIZE) {}
    ~threads_test_c() {}

    void SetUp() {
        for (int i = 0; i < THREAD_POOL_SIZE; i++)
            pool.emplace_back(thread_pool_proc, &queue);
    }
    void TearDown() {}

    void join_threads() {
        for (auto& th : pool)
            if (th.joinable())
                th.join();
    }

    template<int NUM_SLICES, int NUM_B_FRAMES, int NUM_OF_GOPS>
    bool test_flush(int timeout) {
        auto* ref = immitate_gop<NUM_SLICES, NUM_B_FRAMES>(queue);
        for (int i = 0; i < NUM_OF_GOPS; i++)
            ref = immitate_gop<NUM_SLICES, NUM_B_FRAMES, false>(queue, ref);
        CHECK_TIMEOUT(
            {
                queue.kill();
                join_threads();
            }, timeout);
        return true;
    }

    template<int NUM_SLICES, int NUM_B_FRAMES, int NUM_OF_GOPS>
    bool test_multiple_flushes(int num_flushes, int timeout) {
        for (int i = 0; i < num_flushes; i++) {
            immitate_gop<NUM_SLICES, NUM_B_FRAMES>(queue);
            CHECK_TIMEOUT(queue.flush(), timeout);
        }
        CHECK_TIMEOUT(
            {
                queue.kill();
                join_threads();
            }, timeout);
        return true;
    }

private:
    static void thread_pool_proc(task_queue_c* queue) {
        test_slice_task_c* slice_task = nullptr;
        while (queue->get_task((slice_task_c*&)slice_task) == TASK_QUEUE_SUCCESS) {
            slice_task->execute();
            slice_task->done();
        }
    }
    task_queue_c queue;
    std::vector<std::thread> pool;
};

TEST_F(threads_test_c, test_flush) { EXPECT_TRUE((test_flush<68, 3, 100>(1))); }
TEST_F(threads_test_c, test_multiple_flushes) { EXPECT_TRUE((test_multiple_flushes<68, 3, 100>(100, 1))); }
