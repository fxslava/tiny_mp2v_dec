// Copyright � 2021 Vladislav Ovchinnikov. All rights reserved.

#include <thread>
#include <chrono>
#include <iostream>
#include "sample_args.h"
#include "core/decoder.h"

constexpr int CHUNK_SIZE = 65536;
constexpr int MAX_BUFFER_SIZE = CHUNK_SIZE * 512;

void write_yuv(FILE* fp, frame_c* frame) {
    for (int i = 0; i < 3; i++) {
        uint8_t* plane = frame->get_planes(i);
        for (int y = 0; y < frame->get_display_height(i); y++, plane += frame->get_strides(i))
            fwrite(plane, 1, frame->get_display_width(i), fp);
    }
}

void decode_file(std::string filename, mp2v_decoder_c& dec) {
    FILE* fp = fopen(filename.c_str(), "rb");
    uint8_t* buffer = new uint8_t[MAX_BUFFER_SIZE];
    uint8_t* buf_write = &buffer[0];
    uint8_t* buf_read  = &buffer[0];
    int consumed_bytes = 0;
    int rest_bytes = 0;
    while (1) {
        size_t ret_code = fread(buf_write, 1, CHUNK_SIZE, fp);
        buf_write += ret_code;

        bool end_of_file = feof(fp);
        if (end_of_file) {
            *((uint32_t*)buf_write) = 0xb7010000; // end of sequence code
            ret_code += 4;
            ret_code = (ret_code + 15) & ~15;
        }

        dec.decode(buf_read, ret_code);
        buf_read += ret_code;
        if (end_of_file) break;
    }
    dec.flush();
    delete[] buffer;
    fclose(fp);
}

int main(int argc, char* argv[])
{
    std::string* bitstream_file = nullptr, * output_file = nullptr;
    args_parser cmd_parser({
        { "-v", "Input MPEG2 elementary bitsream file", ARG_TYPE_TEXT, &bitstream_file },
        { "-o", "Output YUV stream", ARG_TYPE_TEXT, &output_file }
        }, argc, argv);

    if (output_file) {
        FILE* fp = fopen(output_file->c_str(), "wb");
        if (bitstream_file && fp) {
            mp2v_decoder_c mp2v_decoder({ 1920, 1088, 2, 10, 8, 1024 * 1024, true }, [fp](frame_c* frame) { write_yuv(fp, frame); });

            const auto start = std::chrono::system_clock::now();

            decode_file(*bitstream_file, mp2v_decoder);

            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start);
            printf("Time = %.2f ms\n", static_cast<double>(elapsed_ms.count()));
        }
        fclose(fp);
    }
}