#include "threads.h"

#define TASKQUEUE_HEAD             0x00010000
#define TASKQUEUE_HEAD_NOWORK      0x00008000
#define TASKQUEUE_HEAD_KILL        0x80000000
#define TASKQUEUE_SLICE_NEXT_TASK  -1

slice_task_c::~slice_task_c() { if (owner) owner->slice_done(); }

int picture_task_c::add_slice_task(slice_task_c *task) {
    int idx = slices_tasks.size();
    slices_tasks.push_back(task);
    task->owner = this;
    return idx;
}

bool picture_task_c::add_dependency(picture_task_c* dependency) {
    if (num_dependencies < MAX_NUM_DEPENDENCIES) {
        dependencies[num_dependencies++] = dependency;
        dependency->add_waiter();
        return true;
    }
    else
        return false;
}

void picture_task_c::wait_for_dependencies() {
    for (int i = 0; i < num_dependencies; i++)
        dependencies[i]->wait_for_completion();
}

void picture_task_c::add_waiter() {
    std::lock_guard<std::mutex> lk(mtx);
    int waiters = num_waiters++;
    if (waiters > 0 && status == PICTURE_FREE)
        status = PICTURE_BUSY;
}

void picture_task_c::wait_for_free() {
    if (status != PICTURE_FREE) {
        std::unique_lock<std::mutex> lk(mtx);
        cv_free.wait(lk, [this] { return status == PICTURE_FREE; });
    }
}

void picture_task_c::wait_for_completion() {
    if (status != PICTURE_DONE) {
        std::unique_lock<std::mutex> lk(mtx);
        cv_completed.wait(lk, [this] { return status == PICTURE_DONE; });
    }
}

void picture_task_c::release_waiter() {
    bool is_free = false;
    {
        std::lock_guard<std::mutex> lk(mtx);
        int n_waiters = --num_waiters;
        is_free = !n_waiters;
        if (is_free) status = PICTURE_FREE;
    }
    if (is_free) cv_free.notify_one();
}

void picture_task_c::slice_done() {
    bool pic_done = false;
    {
        std::lock_guard<std::mutex> lk(mtx);
        int done_slices_ = ++done_slices;
        pic_done = (done_slices_ == slices_tasks.size());
        if (pic_done) {
            status = (non_referenceable ? PICTURE_FREE : PICTURE_DONE);
            owner->completed_tasks++;
        }
    }
    if (pic_done) {
        for (int i = 0; i < num_dependencies; i++)
            dependencies[i]->release_waiter();
        cv_completed.notify_one();
    }
}

int task_queue_c::get_pic_idx(int head_) { return head_ >> 17; }

int task_queue_c::get_slice_idx(int head_) { return ((head_ << 16) >> 16); }

int task_queue_c::make_head(int num_slices, int task_idx) {
    return (task_idx << 17) + TASKQUEUE_HEAD + num_slices;
}

bool task_queue_c::wait_for_next_task() {
    while (1) {
        auto ready = ready_to_go_tasks.load();
        if (ready < 0) return false; // Flush condition
        if (ready > 0) break;
    }
    return true;
}

void task_queue_c::next_task(int pic_idx) {
    bool no_work = false;
    {
        std::lock_guard<std::mutex> lk(mtx);
        no_work = (--ready_to_go_tasks == 0);
    }
    if (no_work) cv_no_work.notify_one();
    int new_pic_idx = (pic_idx + 1) % task_queue.size();
    task_queue[new_pic_idx].wait_for_dependencies();
    head.store(make_head(task_queue[new_pic_idx].slices_tasks.size(), new_pic_idx));
}

task_queue_c::task_queue_c(int size) :
    task_queue(size),
    completed_tasks(size),
    ready_to_go_tasks(0),
    head(TASKQUEUE_HEAD | TASKQUEUE_HEAD_NOWORK)
{
    for (auto& task : task_queue)
        task.owner = this;
}

task_status_e task_queue_c::get_task(slice_task_c*& slice_task) {
    slice_task = nullptr;
    while (1) {
        auto head_desc = head.load();
        int  slice_idx = get_slice_idx(head_desc);

        if (slice_idx >= 0) {
            head_desc = --head;
            slice_idx = get_slice_idx(head_desc);
            int pic_idx = get_pic_idx(head_desc);

            if (head_desc & TASKQUEUE_HEAD_KILL) return TASK_QUEUE_KILL;

            if (slice_idx >= 0) {
                slice_task = task_queue[pic_idx].slices_tasks[slice_idx];
                return TASK_QUEUE_SUCCESS;
            }
            else if (slice_idx == TASKQUEUE_SLICE_NEXT_TASK) {
                if (wait_for_next_task())
                    next_task(pic_idx);
            }
        }
        else
            if (head_desc & TASKQUEUE_HEAD_KILL) return TASK_QUEUE_KILL;
    }
}

picture_task_c& task_queue_c::create_task() {
    auto& task_place = task_queue[head_to_work++];
    task_place.wait_for_free();
    return task_place;
}

void task_queue_c::add_task(picture_task_c& task, bool non_referenceable) {
    task.status = PICTURE_BUSY;
    task.non_referenceable = non_referenceable;
    ready_to_go_tasks++;
}

void task_queue_c::flush() {
    if (ready_to_go_tasks.load() > 0)
    {
        std::unique_lock<std::mutex> lk(mtx);
        cv_no_work.wait(lk, [this]() { return ready_to_go_tasks.load() == 0; });
    }
    ready_to_go_tasks.store(-1);
    for (auto& task : task_queue)
        task.wait_for_free();
    head |= TASKQUEUE_HEAD_KILL;
}

void task_queue_c::start() {
    head.store(make_head(task_queue[0].slices_tasks.size(), 0));
}
