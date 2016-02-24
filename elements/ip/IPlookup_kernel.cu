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
#include <nba/engines/cuda/utils.hh>
#include "IPlookup_kernel.hh"

#define IGNORED_IP 0xFFffFFffu

#include <nba/framework/datablock_shared.hh>

extern "C" {

/* The index is given by the order in get_used_datablocks(). */
#define dbid_ipv4_dest_addrs_d     (0)
#define dbid_ipv4_lookup_results_d (1)

__device__ uint32_t ntohl(uint32_t n)
{
    return ((n & 0xff000000) >> 24) | ((n & 0x00ff0000) >> 8) | \
           ((n & 0x0000ff00) << 8)  | ((n & 0x000000ff) << 24);
}

/* The GPU kernel. */
__global__ void ipv4_route_lookup_cuda(
        struct datablock_kernel_arg **datablocks,
        uint32_t count, uint8_t *batch_ids, uint16_t *item_ids,
        uint8_t *checkbits_d,
        uint16_t* __restrict__ TBL24_d,
        uint16_t* __restrict__ TBLlong_d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count) {
        uint8_t batch_idx = batch_ids[idx];
        uint16_t item_idx = item_ids[idx];
        struct datablock_kernel_arg *db_dest_addrs = datablocks[dbid_ipv4_dest_addrs_d];
        struct datablock_kernel_arg *db_results    = datablocks[dbid_ipv4_lookup_results_d];
        uint32_t daddr = ((uint32_t*) db_dest_addrs->batches[batch_idx].buffer_bases_in)[item_idx];
        uint16_t *lookup_result = &((uint16_t *)db_results->batches[batch_idx].buffer_bases_out)[item_idx];

        if (daddr == IGNORED_IP) {
            *lookup_result = 0;
        } else {
            daddr = ntohl(daddr);
            uint16_t temp_dest = TBL24_d[daddr >> 8];
            if (temp_dest & 0x8000u) {
                int index2 = (((uint32_t)(temp_dest & 0x7fff)) << 8) + (daddr & 0xff);
                temp_dest = TBLlong_d[index2];
            }
            *lookup_result = temp_dest;
        }
    }

    __syncthreads();
    if (threadIdx.x == 0 && checkbits_d != NULL) {
        checkbits_d[blockIdx.x] = 1;
    }
}

}

void *nba::ipv4_route_lookup_get_cuda_kernel() {
    return reinterpret_cast<void *> (ipv4_route_lookup_cuda);
}

// vim: ts=8 sts=4 sw=4 et
