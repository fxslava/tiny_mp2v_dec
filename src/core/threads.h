#pragma once
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <functional>
#include <condition_variable>

constexpr int MAX_NUM_DEPENDENCIES = 2;

enum task_status_e {
    TASK_QUEUE_SUCCESS = 0,
    TASK_QUEUE_KILL
};

enum queue_status_e {
    QUEUE_SUSPENDED,
    QUEUE_WORK
};

class picture_task_c;
class task_queue_c;

class slice_task_c {
protected:
    friend class picture_task_c;
public:
    picture_task_c* owner = nullptr;
    virtual bool done();
};

class picture_task_c {
public:
    picture_task_c() : done_slices(0), num_waiters(0) {}
    int add_slice_task(slice_task_c *task);
    bool add_dependency(picture_task_c* dependency);
    void wait_for_dependencies();
    virtual void reset();

protected:
    picture_task_c* dependencies[MAX_NUM_DEPENDENCIES] = {};
    int num_dependencies = 0;

private:
    friend class slice_task_c;
    friend class task_queue_c;

    void add_waiter();
    void wait_for_free();
    void wait_for_completion();
    void release_waiter();
    bool slice_done();

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
    task_queue_c(int size, std::function<picture_task_c*()> constructor);
    task_status_e get_task(slice_task_c*& slice_task);
    picture_task_c* create_task();
    void add_task(picture_task_c* task, bool non_referenceable = false);
    void flush();
    void kill();

private:
    friend class picture_task_c;

    queue_status_e status = QUEUE_SUSPENDED;
    std::atomic<int> ready_to_go_tasks;
    std::atomic<int> head;
    std::vector<picture_task_c*> task_queue;
    std::mutex mtx;
    int head_to_work = 0;

    static int get_pic_idx(int head_);
    static int get_slice_idx(int head_);
    static int make_head(int num_slices, int task_idx);
    bool wait_for_next_task();
    void next_task(int pic_idx);
};
