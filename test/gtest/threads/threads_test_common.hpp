#include "core/threads.h"

class test_slice_task_c : public slice_task_c {
public:
    void execute() {
        auto task_start = std::chrono::high_resolution_clock::now();
        auto task_end = std::chrono::high_resolution_clock::now();
        while (task_end - task_start < std::chrono::microseconds(100))
            task_end = std::chrono::high_resolution_clock::now();
    }
};

template<int NUM_SLICES>
void add_slices(picture_task_c& frame_task) {
    for (int i = 0; i < NUM_SLICES; i++)
        frame_task.add_slice_task(new test_slice_task_c());
}

template<int NUM_SLICES>
void create_p_frame(picture_task_c& frame_task, picture_task_c* ref) {
    frame_task.add_dependency(ref);
    add_slices<NUM_SLICES>(frame_task);
}

template<int NUM_SLICES>
void create_b_frame(picture_task_c& frame_task, picture_task_c* l0, picture_task_c* l1) {
    frame_task.add_dependency(l0);
    frame_task.add_dependency(l1);
    add_slices<NUM_SLICES>(frame_task);
}

template<int NUM_SLICES, int NUM_B_FRAMES, bool FIRST = true>
picture_task_c* immitate_gop(task_queue_c& queue, picture_task_c* ref = nullptr) {
    picture_task_c* _ref = ref;
    if (FIRST) {
        auto& i_frame = queue.create_task();
        add_slices<NUM_SLICES>(i_frame);
        queue.add_task(i_frame);
        _ref = &i_frame;
    }
    auto& p_frame = queue.create_task();
    create_p_frame<NUM_SLICES>(p_frame, _ref);
    queue.add_task(p_frame);
    for (int i = 0; i < NUM_B_FRAMES; i++) {
        auto& b_frame = queue.create_task();
        create_b_frame<NUM_SLICES>(b_frame, _ref, &p_frame);
        queue.add_task(b_frame, true);
    }
    return &p_frame;
}

#define CHECK_TIMEOUT(stmt, timeout) {\
    std::promise<bool> completed; \
    auto stmt_future = completed.get_future(); \
    std::thread([&](std::promise<bool>& completed) { \
        stmt; \
        completed.set_value(true); \
    }, std::ref(completed)).detach(); \
    if (stmt_future.wait_for(std::chrono::seconds(timeout)) == std::future_status::timeout) \
        return false; \
}
