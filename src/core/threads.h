#pragma once
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

constexpr int MAX_NUM_DEPENDENCIES = 2;

enum task_status_e {
    TASK_QUEUE_SUCCESS = 0,
    TASK_QUEUE_KILL
};

enum pic_status_e {
    PICTURE_BUSY,
    PICTURE_DONE,
    PICTURE_FREE
};

class picture_task_c;
class task_queue_c;

class slice_task_c {
private:
    friend class picture_task_c;
    picture_task_c* owner = nullptr;
public:
    ~slice_task_c();
};

class picture_task_c {
public:
    picture_task_c() : done_slices(0), num_waiters(0) {}
    int add_slice_task(slice_task_c *task);
    bool add_dependency(picture_task_c* dependency);
    void wait_for_dependencies();

private:
    friend class slice_task_c;
    friend class task_queue_c;

    void add_waiter();
    void wait_for_free();
    void wait_for_completion();
    void release_waiter();
    void slice_done();

    pic_status_e status = PICTURE_FREE;
    picture_task_c* dependencies[MAX_NUM_DEPENDENCIES] = {};
    int num_dependencies = 0;
    bool non_referenceable = false;
    std::atomic<int> done_slices;
    std::atomic<int> num_waiters;
    std::vector<slice_task_c*> slices_tasks;
    std::condition_variable cv_completed;
    std::condition_variable cv_free;
    std::mutex mtx;
    task_queue_c* owner = {};
};

class task_queue_c {
public:
    task_queue_c(int size);
    task_status_e get_task(slice_task_c*& slice_task);
    picture_task_c& create_task();
    void add_task(picture_task_c& task, bool non_referenceable = false);
    void flush();
    void start();

private:
    friend class picture_task_c;
    std::atomic<int> ready_to_go_tasks;
    std::atomic<int> completed_tasks;
    std::atomic<int> head;
    int head_to_work = 0;
    std::vector<picture_task_c> task_queue;
    std::condition_variable cv_no_work;
    std::mutex mtx;

    static int get_pic_idx(int head_);
    static int get_slice_idx(int head_);
    static int make_head(int num_slices, int task_idx);
    bool wait_for_next_task();
    void next_task(int pic_idx);
};
