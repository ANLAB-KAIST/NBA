#ifndef __NSHADER_ENGINES_CUDA_COMPAT_HH__
#define __NSHADER_ENGINES_CUDA_COMPAT_HH__

#define NSHADER_MAX_COPROC_PPDEPTH (256u)

struct datablock_kernel_arg {
    uint32_t total_item_count_in;
    uint32_t total_item_count_out;
    void *buffer_bases_in[NSHADER_MAX_COPROC_PPDEPTH];
    void *buffer_bases_out[NSHADER_MAX_COPROC_PPDEPTH];
    uint32_t item_count_in[NSHADER_MAX_COPROC_PPDEPTH];
    uint32_t item_count_out[NSHADER_MAX_COPROC_PPDEPTH];
    union {
        uint16_t item_size_in;
        uint16_t *item_sizes_in[NSHADER_MAX_COPROC_PPDEPTH];
    };
    union {
        uint16_t item_size_out;
        uint16_t *item_sizes_out[NSHADER_MAX_COPROC_PPDEPTH];
    };
    uint16_t *item_offsets_in[NSHADER_MAX_COPROC_PPDEPTH];
    uint16_t *item_offsets_out[NSHADER_MAX_COPROC_PPDEPTH];
};

#endif
