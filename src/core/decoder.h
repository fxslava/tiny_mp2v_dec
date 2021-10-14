// Copyright � 2021 Vladislav Ovchinnikov. All rights reserved.
#pragma once
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "common/queue.hpp"
#include "mp2v_hdr.h"
#include "api/bitstream.h"
#include "mb_decoder.h"

#define MP2V_MT

constexpr int MAX_NUM_THREADS = 256;
constexpr int MAX_B_FRAMES = 8;
constexpr int CACHE_LINE = 64;

class mp2v_picture_c;
class mp2v_decoder_c;

struct slice_task_t {
    macroblock_context_cache_t task_cache;
    bitstream_reader_c bs;
};

struct decoder_config_t {
    int width;
    int height;
    int chroma_format;
    int frames_pool_size;
    int pictures_pool_size;
    int num_threads;
    bool reordering;
};

class frame_c {
    friend class mp2v_picture_c;
public:
    frame_c(int width, int height, int chroma_format);
    ~frame_c();

    uint8_t* get_planes (int plane_idx) { return m_planes[plane_idx]; }
    int      get_strides(int plane_idx) { return m_stride[plane_idx]; }
    int      get_width  (int plane_idx) { return m_width [plane_idx]; }
    int      get_height (int plane_idx) { return m_height[plane_idx]; }
private:
    uint32_t m_width [3] = { 0 };
    uint32_t m_height[3] = { 0 };
    uint32_t m_stride[3] = { 0 };
    uint8_t* m_planes[3] = { 0 };
};

class mp2v_picture_c {
public:
    mp2v_picture_c(bitstream_reader_c* bitstream, mp2v_decoder_c* decoder, frame_c* frame) : m_bs(bitstream), m_dec(decoder), m_frame(frame) {};
    void init();
    void attach(frame_c* frame) { m_frame = frame; }
    template <bool weak = false> bool decode_slice();
    frame_c* get_frame() { return m_frame; }

private:
    bitstream_reader_c* m_bs;
    mp2v_decoder_c* m_dec;
    uint8_t quantiser_matrices[4][64];
    parse_macroblock_func_t m_parse_macroblock_func = nullptr;
    frame_c* m_frame;

public:
    // headers
    picture_header_t m_picture_header = { 0 }; //mandatory
    picture_coding_extension_t m_picture_coding_extension = { 0 }; //mandatory
    quant_matrix_extension_t* m_quant_matrix_extension = nullptr;
    copyright_extension_t* m_copyright_extension = nullptr;
    picture_display_extension_t* m_picture_display_extension = nullptr;
    picture_spatial_scalable_extension_t* m_picture_spatial_scalable_extension = nullptr;
    picture_temporal_scalable_extension_t* m_picture_temporal_scalable_extension = nullptr;

#ifdef MP2V_MT
    std::vector<slice_task_t> picture_slices_tasks;
    bool decode_task(int task_id);
    void reset_task_list() { picture_slices_tasks.clear(); }
#endif
};

class mp2v_decoder_c {
    friend class mp2v_picture_c;
public:
    mp2v_decoder_c() : m_frames_pool(100), m_output_frames(100), num_slices_for_work(0), num_slices_done(0) {};
    ~mp2v_decoder_c();
    bool decoder_init(decoder_config_t* config);
    bool decode(uint8_t* buffer, int len);
    void get_decoded_frame(frame_c*& frame) { m_output_frames.pop(frame); }
    void release_frame(frame_c* frame) { m_frames_pool.push(frame); }

protected:
    bool decode_user_data();
    bool decode_extension_data(mp2v_picture_c* pic);
    void push_frame(frame_c* frame) { m_output_frames.push(frame); }
    uint32_t get_next_start_code();
    void flush_mini_gop();
    void out_pic(mp2v_picture_c* cur_pic);
    bool reordering = true;
    bitstream_reader_c m_bs;
    frame_c* ref_frames[2] = { 0 };
    ThreadSafeQ<frame_c*> m_frames_pool;
    ThreadSafeQ<frame_c*> m_output_frames;
    std::vector<mp2v_picture_c*> m_pictures_pool;
    std::vector<uint32_t*> start_code_tbl;
    uint32_t  start_code_idx = 0;
#ifdef MP2V_MT
    static void threadpool_task_scheduler(mp2v_decoder_c *dec);
    void flush(mp2v_picture_c* cur_pic, int num_slices);
    mp2v_picture_c* processing_picture = nullptr;
    std::atomic<int> num_slices_for_work;
    std::atomic<int> num_slices_done;
    std::mutex done_mtx;
    std::condition_variable cv_pic_done;
    std::thread* thread_pool[MAX_NUM_THREADS] = { 0 };
    int done_threashold = 0;
    int num_threads = 1;
#endif

public:
    // headers & user data
    std::vector<uint8_t> user_data;
    sequence_header_t m_sequence_header = { 0 }; //mandatory
    sequence_extension_t m_sequence_extension = { 0 }; //mandatory
    sequence_display_extension_t* m_sequence_display_extension = nullptr;
    sequence_scalable_extension_t* m_sequence_scalable_extension = nullptr;
    group_of_pictures_header_t* m_group_of_pictures_header = nullptr;
};
