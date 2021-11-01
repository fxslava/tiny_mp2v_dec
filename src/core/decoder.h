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
#include "bitstream.h"
#include "mb_decoder.h"
#include "threads.h"

//#define MP2V_MT

constexpr int MAX_NUM_THREADS = 256;
constexpr int MAX_B_FRAMES = 8;
constexpr int CACHE_LINE = 64;
constexpr int DEFAULT_BITSTREAM_BUFFER_SIZE = 1024*1024;

class mp2v_picture_c;
class mp2v_decoder_c;

struct decoder_config_t {
    int width;
    int height;
    int chroma_format;
    int pictures_pool_size;
    int num_threads;
    int bitstream_chunk_size;
    bool reordering;
};

class frame_c {
    friend class mp2v_picture_c;
public:
    frame_c(int width, int height, int chroma_format);
    ~frame_c();

    uint8_t* get_planes (int plane_idx) { return m_planes[plane_idx]; }
    int      get_strides(int plane_idx) { return m_stride[plane_idx]; }
    int      get_display_width  (int plane_idx) { return m_display_width [plane_idx]; }
    int      get_display_height (int plane_idx) { return m_display_height[plane_idx]; }
    void     set_display_size   (int display_width, int display_height);
private:
    int m_chroma_format = chroma_format_420;
    uint32_t m_width [3] = { 0 };
    uint32_t m_height[3] = { 0 };
    uint32_t m_display_width [3] = { 0 };
    uint32_t m_display_height[3] = { 0 };
    uint32_t m_stride[3] = { 0 };
    uint8_t* m_planes[3] = { 0 };
};

class mp2v_slice_task_c : public slice_task_c {
public:
    bitstream_reader_c bs;
    void decode();
};

class mp2v_picture_c : public picture_task_c {
    friend class mp2v_decoder_c;
public:
    mp2v_picture_c(mp2v_decoder_c* decoder, frame_c* frame, int bitstream_buffer_size = DEFAULT_BITSTREAM_BUFFER_SIZE) : 
        m_dec(decoder), m_frame(frame), bitstream_buffer(bitstream_buffer_size) {
        cur_bistream_pos = &bitstream_buffer[0];
    };
    void init();
    void attach(frame_c* frame) { m_frame = frame; }
    bool decode_slice(bitstream_reader_c bs);
    frame_c* get_frame() { return m_frame; }
    void reset() {
        picture_task_c::reset();
        cur_bistream_pos = &bitstream_buffer[0];
        last_start_code = nullptr;
    }

private:
    uint8_t* last_start_code = nullptr;
    uint8_t* cur_bistream_pos = nullptr;
    std::vector<uint8_t> bitstream_buffer;
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
    mp2v_decoder_c()
#ifndef MP2V_MT
        : m_done_pics(100), m_free_pics(100)
#endif
    {};
    mp2v_decoder_c(const decoder_config_t& config, std::function<void(frame_c*)> renderer)
#ifndef MP2V_MT
        : m_done_pics(100), m_free_pics(100)
#endif
    {
        decoder_init(config, renderer);
    };
    ~mp2v_decoder_c();
    bool decoder_init(const decoder_config_t& config, std::function<void(frame_c*)> renderer);
    void decode(uint8_t* buffer, int len);
    void decode_unit(uint8_t* start_code_ptr);
    void flush();

protected:
    bool decode_user_data();
    bool decode_extension_data(mp2v_picture_c* pic);
    mp2v_picture_c* new_pic();
    void out_pic(mp2v_picture_c* cur_pic);
    bool reordering = true;
    bitstream_reader_c m_bs;
    mp2v_picture_c* ref_frames[2] = { 0 };
    std::function<void(frame_c*)> render_func;
    std::thread* render_thread = nullptr;
    static void decoder_output_scheduler(mp2v_decoder_c* dec);
#ifdef MP2V_MT
    static void threadpool_task_scheduler(mp2v_decoder_c *dec);
    std::thread* thread_pool[MAX_NUM_THREADS] = { 0 };
    task_queue_c* task_queue = nullptr;
#else
    ThreadSafeQ<mp2v_picture_c*> m_done_pics;
    ThreadSafeQ<mp2v_picture_c*> m_free_pics;
    std::vector<mp2v_picture_c*> m_pictures_pool;
#endif
    // Decoder state variables
    uint8_t* prev_start_code = nullptr;
    uint8_t* last_start_code = nullptr;
    mp2v_picture_c* cur_pic = nullptr;
    bool new_picture = false;

public:
    // headers & user data
    std::vector<uint8_t> user_data;
    sequence_header_t m_sequence_header = { 0 }; //mandatory
    sequence_extension_t m_sequence_extension = { 0 }; //mandatory
    sequence_display_extension_t* m_sequence_display_extension = nullptr;
    sequence_scalable_extension_t* m_sequence_scalable_extension = nullptr;
    group_of_pictures_header_t* m_group_of_pictures_header = nullptr;
};
