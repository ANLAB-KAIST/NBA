#ifndef __NSHADER_OBJTYPES_HH__
#define __NSHADER_OBJTYPES_HH__

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

#endif

// vim: ts=8 sts=4 sw=4 et
