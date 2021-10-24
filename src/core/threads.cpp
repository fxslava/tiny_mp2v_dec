#include "threads.h"
#include <algorithm>

#define TASKQUEUE_HEAD             0x00010000
#define TASKQUEUE_HEAD_NOWORK      0x0000fffe
#define TASKQUEUE_HEAD_KILL        0x80000000
#define TASKQUEUE_SLICE_NEXT_TASK  -1

bool slice_task_c::done() { 
    if (owner) 
        return owner->slice_done(); 
    return false;
}

int picture_task_c::add_slice_task(slice_task_c *task) {
    int idx = slices_tasks.size();
    slices_tasks.push_back(task);
    task->owner = this;
    return idx;
}

bool picture_task_c::add_dependency(picture_task_c* dependency) {
    if (num_dependencies < MAX_NUM_DEPENDENCIES) {
        dependencies[num_dependencies++] = dependency;
        if (dependency) dependency->add_waiter();
        return true;
    }
    else
        return false;
}

void picture_task_c::wait_for_dependencies() {
    for (int i = 0; i < num_dependencies; i++)
        if (dependencies[i])
            dependencies[i]->wait_for_completion();
}

void picture_task_c::reset() {
    num_dependencies = 0;
    for (auto*& dep : dependencies) dep = nullptr;
    non_referenceable = false;
    done_slices.store(0);
    num_waiters.store(0);
    slices_tasks.clear();
}

void picture_task_c::add_waiter() { num_waiters++; }

void picture_task_c::wait_for_free() {
    std::unique_lock<std::mutex> lk(mtx);
    cv_free.wait(lk, [this] { return num_waiters == 0; });
}

void picture_task_c::wait_for_completion() {
    std::unique_lock<std::mutex> lk(mtx);
    cv_completed.wait(lk, [this] { return done_slices.load() == slices_tasks.size(); });
}

void picture_task_c::release_waiter() {
    bool is_free = false;
    {
        std::lock_guard<std::mutex> lk(mtx);
        int n_waiters = --num_waiters;
        is_free = !n_waiters;
    }
    if (is_free) cv_free.notify_all();
}

bool picture_task_c::slice_done() {
    bool pic_done = false;
    {
        std::lock_guard<std::mutex> lk(mtx);
        int done_slices_ = ++done_slices;
        pic_done = (done_slices_ >= slices_tasks.size());
    }
    if (pic_done) {
        for (int i = 0; i < num_dependencies; i++)
            if (dependencies[i])
                dependencies[i]->release_waiter();
        owner->pic_done();
        cv_completed.notify_all();
        if (non_referenceable)
            cv_free.notify_all();
    }
    return pic_done;
}

int task_queue_c::get_pic_idx(int head_) { return head_ >> 17; }

int task_queue_c::get_slice_idx(int head_) { return ((head_ << 16) >> 16); }

int task_queue_c::make_head(int num_slices, int task_idx) {
    return (task_idx << 17) + TASKQUEUE_HEAD + num_slices;
}

bool task_queue_c::wait_for_next_task() {
    while (1) {
        auto ready = ready_to_go_tasks.load();
        if (ready < 0) {
            head.store(TASKQUEUE_HEAD | TASKQUEUE_HEAD_KILL | TASKQUEUE_HEAD_NOWORK);
            return false;
        }
        if (ready > 0) break;
    }
    return true;
}

void task_queue_c::next_task(int pic_idx) {
    ready_to_go_tasks--;
    int new_pic_idx = (pic_idx + 1) % task_queue.size();
    task_queue[new_pic_idx]->wait_for_dependencies();
    int new_head = make_head(task_queue[new_pic_idx]->slices_tasks.size(), new_pic_idx);
    head.store(new_head);
}

void task_queue_c::pic_done() {
    int num_done = ++done_tasks;
    if (num_done)
        cv_done.notify_one();
}

task_queue_c::task_queue_c(int size, std::function<picture_task_c* ()> constructor) :
    task_queue(size),
    done_tasks(0),
    ready_to_go_tasks(0),
    head(TASKQUEUE_HEAD | TASKQUEUE_HEAD_NOWORK)
{
    std::generate(task_queue.begin(), task_queue.end(), constructor);
    for (auto* task : task_queue) task->owner = this;
}

task_status_e task_queue_c::get_task(slice_task_c*& slice_task) {
    slice_task = nullptr;
    while (1) {
        int head_desc = head.load();
        if (head_desc & TASKQUEUE_HEAD_KILL) return TASK_QUEUE_KILL;
        int slice_idx = get_slice_idx(head_desc);
        if (slice_idx >= 0) {
            head_desc = --head;
            slice_idx = get_slice_idx(head_desc);
            int pic_idx = get_pic_idx(head_desc);

            if (slice_idx >= 0) {
                slice_task = task_queue[pic_idx]->slices_tasks[slice_idx];
                return TASK_QUEUE_SUCCESS;
            }
            else if (slice_idx == TASKQUEUE_SLICE_NEXT_TASK) {
                if (wait_for_next_task())
                    next_task(pic_idx);
            }
        }
    }
}

picture_task_c* task_queue_c::create_task() {
    auto& task_place = task_queue[head_to_work++];
    head_to_work %= task_queue.size();
    task_place->wait_for_completion();
    task_place->wait_for_free();
    task_place->reset();
    return task_place;
}

picture_task_c* task_queue_c::get_decoded() {
    auto& task_place = task_queue[tail_decoded++];
    task_place->wait_for_completion();
    tail_decoded %= task_queue.size();
    if (!done_tasks) {
        std::unique_lock<std::mutex> lk(mtx);
        cv_done.wait(lk, [this] { return (done_tasks > 0) || (done_tasks < 0); });
    }
    if (done_tasks < 0) return nullptr;
    int done_pics = --done_tasks;
    if (!done_pics)
        cv_no_done.notify_one();
    return task_place;
}

void task_queue_c::add_task(picture_task_c* task, bool non_referenceable) {
    task->non_referenceable = non_referenceable;
    if (status == QUEUE_SUSPENDED) {
        head.store(make_head(task_queue[0]->slices_tasks.size(), 0));
        status = QUEUE_WORK;
    }
    else
        ready_to_go_tasks++;
}

void task_queue_c::flush() {
    for (auto* task : task_queue) {
        task->wait_for_completion();
        task->wait_for_free();
        task->reset();
    }
}

void task_queue_c::kill() {
    flush();
    ready_to_go_tasks.store(-1);

    if (!done_tasks) {
        std::unique_lock<std::mutex> lk(mtx);
        cv_no_done.wait(lk, [this] { return done_tasks.load() == 0; });
    }
    done_tasks.store(-1);
    cv_done.notify_one();
}
