#include "bitstream.h"
#include "mp2v_vlc.h"
#include "scan.h"
#include <stdint.h>

template <typename T, int count>
static void local_copy_array(bitstream_reader_c* bs, T* dst) {
    for (int i = 0; i < count; i++) {
        dst[i] = bs->read_next_bits(sizeof(T) * 8);
    }
}
