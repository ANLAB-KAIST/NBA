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

#ifdef USE_KNAPP
struct knapp_memobj {
    uint32_t buffer_id;
    void *unwrap_ptr;
};
#endif

/* Common object types */
typedef union {
    void *ptr;
    #ifdef USE_PHI
    cl_mem clmem;
    #endif
    #ifdef USE_KNAPP
    struct knapp_memobj m;
    #endif
} dev_mem_t;

typedef union {
    void *ptr;
    #ifdef USE_KNAPP
    struct knapp_memobj m;
    #endif
} host_mem_t;

typedef union {
    void *ptr;
    #ifdef USE_PHI
    cl_kernel clkernel;
    #endif
    #ifdef USE_KNAPP
    int kernel_id;
    #endif
} dev_kernel_t;

typedef union {
    void *ptr;
    #ifdef USE_CUDA
    cudaEvent_t cuev;
    #endif
    #ifdef USE_PHI
    cl_event clev;
    #endif
} dev_event_t;

struct resource_param {
    uint32_t num_workitems;
    uint32_t num_workgroups;
    uint32_t num_threads_per_workgroup;
    uint32_t task_id; // FIXME: refactor
};

struct kernel_arg {
    void *ptr;
    size_t size;
    size_t align;
};

enum io_direction_hint : int {
    AGNOSTIC       = 0,
    HOST_TO_DEVICE = 1,
    DEVICE_TO_HOST = 2,
};


#endif

// vim: ts=8 sts=4 sw=4 et
