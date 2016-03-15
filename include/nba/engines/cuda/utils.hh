#ifndef __CUDA_UTILS_HH__
#define __CUDA_UTILS_HH__

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

/*
 * NVIDIA does not guarantee compatibility of the cutil library
 * since it is intended for the example purposes only.
 * We should have our own cutilSafeCall() macro.
 */

extern "C" {

inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
    if (cudaSuccess == err || cudaErrorCudartUnloading == err)
        return;
    fprintf(stderr, "%s(%i): CUDA Runtime Error %d: %s.\n",
            file, line, (int)err, cudaGetErrorString(err));
    exit(-1);
}

}

#define cutilSafeCall(err)       __cudaSafeCall(err, __FILE__, __LINE__)

#endif // __CUDA_UTILS_HH__

// vim: ts=8 sts=4 sw=4 et
