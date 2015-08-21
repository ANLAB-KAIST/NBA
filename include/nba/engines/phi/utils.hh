#ifndef __PHI_UTILS_HH__
#define __PHI_UTILS_HH__

#include <stdio.h>
#include <stdlib.h>
#include <CL/opencl.h>

/*
 * NVIDIA does not guarantee compatibility of the cutil library
 * since it is intended for the example purposes only.
 * We should have our own cutilSafeCall() macro.
 */

#ifdef __cplusplus

#endif
inline void __phiSafeCall(cl_int ret, const char *file, const int line)
{
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "%s(%i): Phi OpenCL Runtime Error %d: (TODO: user-friendly str).\n",
            file, line, ret);
        exit(-1);
    
}

#ifdef __cplusplus
}
#endif

#define phiSafeCall(err)         __phiSafeCall(err, __FILE__, __LINE__)

#endif // __CUDA_UTILS_HH__

// vim: ts=8 sts=4 sw=4 et
