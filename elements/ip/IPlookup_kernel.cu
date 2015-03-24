#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

// includes, project
#include <cuda.h>
#include "../../engines/cuda/utils.hh"
#include "IPlookup_kernel.hh"

#define IGNORED_IP 0xFFffFFffu

extern "C" {

/* The GPU kernel. */
__global__ void ipv4_route_lookup_cuda(
        uint32_t* __restrict__ dest_addrs_d,
        uint16_t *results_d,
        size_t *input_size_arr,
        size_t *output_size_arr,
        uint32_t N,
        uint8_t *checkbits_d,
        uint16_t* __restrict__ TBL24_d,
        uint16_t* __restrict__ TBLlong_d)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    uint16_t temp_dest;

    if (index < N) {
        uint32_t daddr = dest_addrs_d[index];

        if (daddr != IGNORED_IP) {

            temp_dest = TBL24_d[daddr >> 8];

            if (temp_dest & 0x8000u) {
                int index2 = (((uint32_t)(temp_dest & 0x7fff)) << 8) + (daddr & 0xff);
                temp_dest = TBLlong_d[index2];
            }

            results_d[index] = temp_dest;
        }
    }

    // TODO: wrap CUDA function and move to framework-side.
    __syncthreads();
    if (threadIdx.x == 0 && checkbits_d != NULL) {
        *(checkbits_d + blockIdx.x) = 1;
    }
}

}

void *nba::ipv4_route_lookup_get_cuda_kernel() {
    return reinterpret_cast<void *> (ipv4_route_lookup_cuda);
}

// vim: ts=8 sts=4 sw=4 et
