#ifndef __NBA_OFFLOADTYPES_HH__
#define __NBA_OFFLOADTYPES_HH__

#include <cstdlib>
#include <cstdint>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
#ifdef USE_PHI
#include <CL/opencl.h>
#endif

/* Common object types */
typedef union memobj {
    void *ptr;
    #ifdef USE_PHI
    cl_mem clmem;
    #endif
} memory_t;

typedef union kernelobj {
    void *ptr;
    #ifdef USE_PHI
    cl_kernel clkernel;
    #endif
} kernel_t;

typedef union eventobj {
    void *ptr;
    #ifdef USE_CUDA
    cudaEvent_t cuev;
    #endif
    #ifdef USE_PHI
    cl_event clev;
    #endif
} event_t;

struct resource_param {
    uint32_t num_workitems;
    uint32_t num_workgroups;
    uint32_t num_threads_per_workgroup;
};

struct kernel_arg {
    void *ptr;
    size_t size;
    size_t align;
};

enum io_direction_hint {
    HOST_TO_DEVICE = 0,
    DEVICE_TO_HOST = 1,
};


#endif

// vim: ts=8 sts=4 sw=4 et
