#ifndef __NBA_DATABLOCK_SHARED_HH__
#define __NBA_DATABLOCK_SHARED_HH__

/*
 * This header is included by both .cc/.cu sources.
 * Note that the nvcc should support C++11 (CUDA v6.5 or higher).
 */

#include <cstdint>
#include <nba/core/shiftedint.hh>

struct alignas(8) datablock_batch_info {
    void *buffer_bases;
    uint32_t item_count;
    uint16_t *item_sizes;
    nba::dev_offset_t *item_offsets;
};

struct alignas(8) datablock_kernel_arg {
    uint32_t total_item_count;
    uint16_t item_size;  // for fixed-size cases
    struct datablock_batch_info batches[0];
};

#endif
