#ifndef __NBA_ENGINES_CUDA_COMPAT_HH__
#define __NBA_ENGINES_CUDA_COMPAT_HH__

#define NBA_MAX_COPROC_PPDEPTH (256u)

struct datablock_kernel_arg {
    uint32_t total_item_count_in;
    uint32_t total_item_count_out;
    void *buffer_bases_in[NBA_MAX_COPROC_PPDEPTH];
    void *buffer_bases_out[NBA_MAX_COPROC_PPDEPTH];
    uint32_t item_count_in[NBA_MAX_COPROC_PPDEPTH];
    uint32_t item_count_out[NBA_MAX_COPROC_PPDEPTH];
    union {
        uint16_t item_size_in;
        uint16_t *item_sizes_in[NBA_MAX_COPROC_PPDEPTH];
    };
    union {
        uint16_t item_size_out;
        uint16_t *item_sizes_out[NBA_MAX_COPROC_PPDEPTH];
    };
    uint16_t *item_offsets_in[NBA_MAX_COPROC_PPDEPTH];
    uint16_t *item_offsets_out[NBA_MAX_COPROC_PPDEPTH];
};

#endif
