#ifndef __NBA_ENGINES_CUDA_COMPAT_HH__
#define __NBA_ENGINES_CUDA_COMPAT_HH__

/*
 * This header is included by .cu sources.
 * We put only relevant data structures here for use in CUDA codes.
 * Note that the nvcc should support C++11 (CUDA v6.5 or higher).
 */

#include <nba/framework/config.hh>

struct datablock_batch_info {
    void *buffer_bases_in;
    void *buffer_bases_out;
    uint32_t item_count_in;
    uint32_t item_count_out;
    uint16_t *item_sizes_in;
    uint16_t *item_sizes_out;
    uint32_t *item_offsets_in;
    uint32_t *item_offsets_out;
}; // __cuda_aligned

struct datablock_kernel_arg {
    uint32_t total_item_count_in;
    uint32_t total_item_count_out;
    uint16_t item_size_in;  // for fixed-size cases
    uint16_t item_size_out; // for fixed-size cases
    struct datablock_batch_info batches[0];
}; // __cuda_aligned

#endif
