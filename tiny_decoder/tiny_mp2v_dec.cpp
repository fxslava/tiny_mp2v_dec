// Copyright � 2021 Vladislav Ovchinnikov. All rights reserved.

#include <thread>
#include <chrono>
#include <iostream>
#include "sample_args.h"
#include "core/decoder.h"

std::vector<uint32_t, AlignmentAllocator<uint8_t, 32>> buffer_pool;

void write_yuv(FILE* fp, frame_c* frame) {
    for (int i = 0; i < 3; i++) {
        uint8_t* plane = frame->get_planes(i);
        for (int y = 0; y < frame->get_height(i); y++, plane += frame->get_strides(i))
            fwrite(plane, 1, frame->get_width(i), fp);
    }
}

void load_bitstream(std::string input_file) {
    std::ifstream fp(input_file, std::ios::binary);

    // Calculate size of buffer
    fp.seekg(0, std::ios_base::end);
    std::size_t size = fp.tellg();
    size = ((size + 15) & (~15));
    fp.seekg(0, std::ios_base::beg);

    // Allocate buffer
    buffer_pool.resize(size / sizeof(uint32_t));

    // read file
    fp.read((char*)&buffer_pool[0], size);
    fp.close();
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
            load_bitstream(*bitstream_file);
            mp2v_decoder_c mp2v_decoder({ 1920, 1088, 2, 10, 8, true }, [fp](frame_c* frame) { write_yuv(fp, frame); });

            const auto start = std::chrono::system_clock::now();

            mp2v_decoder.decode((uint8_t*)&buffer_pool[0], buffer_pool.size() * 4);

            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start);
            printf("Time = %.2f ms\n", static_cast<double>(elapsed_ms.count()));
        }
        fclose(fp);
    }
}