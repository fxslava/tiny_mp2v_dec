// Copyright � 2021 Vladislav Ovchinnikov. All rights reserved.
#pragma once
#include <deque>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "common/queue.hpp"
#include "mp2v_hdr.h"
#include "api/bitstream.h"
#include "mb_decoder.h"
#include "threads.h"

//#define MP2V_MT

constexpr int MAX_NUM_THREADS = 256;
constexpr int MAX_B_FRAMES = 8;
constexpr int CACHE_LINE = 64;

class mp2v_picture_c;
class mp2v_decoder_c;

class mp2v_slice_task_c : public slice_task_c {
public:
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

class mp2v_picture_c : public picture_task_c {
public:
    mp2v_picture_c(mp2v_decoder_c* decoder, frame_c* frame) : m_dec(decoder), m_frame(frame) {};
    void init();
    void attach(frame_c* frame) { m_frame = frame; }
    bool decode_slice(bitstream_reader_c bs);
    frame_c* get_frame() { return m_frame; }

private:
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
};

class mp2v_decoder_c {
    friend class mp2v_picture_c;
public:
    mp2v_decoder_c() : m_frames_pool(100), m_output_frames(100) {};
    ~mp2v_decoder_c();
    bool decoder_init(decoder_config_t* config);
    bool decode(uint8_t* buffer, int len);
    void get_decoded_frame(frame_c*& frame) { m_output_frames.pop(frame); }
    void release_frame(frame_c* frame) { m_frames_pool.push(frame); }

protected:
    bool decode_user_data();
    bool decode_extension_data(mp2v_picture_c* pic);
    void push_frame(frame_c* frame) { m_output_frames.push(frame); }
    void flush_mini_gop();
    mp2v_picture_c* new_pic();
    void out_pic(mp2v_picture_c* cur_pic);
    bool reordering = true;
    bitstream_reader_c m_bs;
    mp2v_picture_c* ref_frames[2] = { 0 };
    ThreadSafeQ<frame_c*> m_frames_pool;
    ThreadSafeQ<frame_c*> m_output_frames;
    std::deque<mp2v_picture_c*> m_pictures_pool;

#ifdef MP2V_MT
    static void threadpool_task_scheduler(mp2v_decoder_c *dec);
    std::thread* thread_pool[MAX_NUM_THREADS] = { 0 };
    task_queue_c* task_queue = nullptr;
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
