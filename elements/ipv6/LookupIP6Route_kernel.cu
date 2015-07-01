#include <stdint.h>
#include <cassert>
#include <cstdio>

#include <cuda.h>
#include "../../engines/cuda/utils.hh"
#include "LookupIP6Route_kernel.hh"

#include "util_jhash.h"
#include "util_hash_table.hh"
#include "../../engines/cuda/compat.hh"

extern "C" {

__device__ u32 gpu_jhash2(const u32 *k, u32 length, u32 initval)
{
    u32 a, b, c, len;

    a = b = JHASH_GOLDEN_RATIO;
    c = initval;
    len = length;

    while (len >= 3) {
        a += k[0];
        b += k[1];
        c += k[2];
        __jhash_mix(a, b, c);
        k += 3; len -= 3;
    }

    c += length * 4;

    switch (len) {
    case 2 : b += k[1];
    case 1 : a += k[0];
    };

    __jhash_mix(a,b,c);

    return c;
}

__device__ u32 gpu_jhash2_optimized(uint32_t k0, uint32_t k1, uint32_t k2, uint32_t k3)
{
    u32 a, b, c;

    a = b = JHASH_GOLDEN_RATIO;
    c = 0;

    a += k0;
    b += k1;
    c += k2;
    __jhash_mix(a, b, c);

    c += 4 * 4;
    a += k3;
    __jhash_mix(a, b, c);

    return c;
}

#define HASH(x, y) (gpu_jhash2((u32*)&x, 4, 0) % y)

__device__ uint32_t hashtable_find(uint64_t ip0, uint64_t ip1, const Item* __restrict__ table, const int tablesize)
{
    uint32_t index = gpu_jhash2_optimized(ip0 >> 32, (uint32_t)ip0, ip1 >> 32, (uint32_t)ip1) % tablesize;

    union { uint32_t u32; uint16_t u16[2]; } ret;
    ret.u32 = 0;

    if (table[index].state != IPV6_HASHTABLE_EMPTY) {
        do {
            if (table[index].key.u64[0] == ip0 &&
                table[index].key.u64[1] == ip1)
            {
                ret.u16[0] = table[index].val;
                ret.u16[1] = table[index].state;
                break;
            }
            index = table[index].next;
        } while (index != 0);
    }

    return ret.u32;
}

__device__ uint32_t gpu_route_lookup_one(uint64_t ip0, uint64_t ip1,
        Item* __restrict__ tables[], size_t* __restrict__ table_sizes)
{
    int start = 0;
    int end = 127;
    uint32_t bmp = 0;

    do {
        uint64_t masked_ip0;
        uint64_t masked_ip1;

        int len = 128 - (start + end) / 2;
        if (len < 64) {
            masked_ip0 = ((ip0 >> len) << len);
            masked_ip1 = ip1;
        } else if (len < 128) {
            masked_ip0 = 0;
            masked_ip1 = ((ip1 >> (len - 64)) << (len - 64));
        } else {
            masked_ip0 = 0;
            masked_ip1 = 0;
        }

        len = 128 - len;

        uint32_t result = hashtable_find(masked_ip0, masked_ip1,
                         tables[len], table_sizes[len]);

        if (result == 0) {
            end = len - 1;
        } else {
            bmp = result;
            start = len + 1;
        }

    } while (start <= end );

    return bmp;
}

#define dbid_ipv6_dest_addrs_d     (0)
#define dbid_ipv6_lookup_results_d (1)
struct _cu_uint128_t
{
	uint64_t ip0;
	uint64_t ip1;
};

__device__ uint64_t ntohll(uint64_t val)
{
        return ( (((val) >> 56) & 0x00000000000000ff) | (((val) >> 40) & 0x000000000000ff00) | \
                (((val) >> 24) & 0x0000000000ff0000) | (((val) >>  8) & 0x00000000ff000000) | \
                (((val) <<  8) & 0x000000ff00000000) | (((val) << 24) & 0x0000ff0000000000) | \
                (((val) << 40) & 0x00ff000000000000) | (((val) << 56) & 0xff00000000000000) );
}

__global__ void ipv6_route_lookup_cuda(
        struct datablock_kernel_arg *datablocks,
        uint32_t count, uint16_t *batch_ids, uint16_t *item_ids,
        uint8_t *checkbits_d,
        Item** __restrict__ tables_d,
        size_t* __restrict__ table_sizes_d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count) {
        uint16_t batch_idx = batch_ids[idx];
        uint16_t item_idx  = item_ids[idx];
        struct datablock_kernel_arg *db_dest_addrs = &datablocks[dbid_ipv6_dest_addrs_d];
        struct datablock_kernel_arg *db_results    = &datablocks[dbid_ipv6_lookup_results_d];
        struct _cu_uint128_t daddr = ((struct _cu_uint128_t*) db_dest_addrs->buffer_bases_in[batch_idx])[item_idx];
        uint16_t *lookup_result = &((uint16_t *) db_results->buffer_bases_out[batch_idx])[item_idx];

        // NOTE: On FERMI devices, using shared memory to store just 128
        //   pointers is not necessary since they have on-chip L1
        //   cache.
        // NOTE: This was the point of bug for "random" CUDA errors.
        //   (maybe due to out-of-bound access to shared memory?)
        // UPDATE: On new NBA with CUDA 5.5, this code does neither seem to
        //         generate any errors nor bring performance benefits.

        uint64_t ip0 = ntohll(daddr.ip1);
        uint64_t ip1 = ntohll(daddr.ip0);
        if (ip0 == 0xffffffffffffffffu && ip1 == 0xffffffffffffffffu) {
            *lookup_result = 0;
        } else {
            *lookup_result = (uint16_t) gpu_route_lookup_one(ip0, ip1, tables_d, table_sizes_d);
        }
    }

    __syncthreads();
    if (threadIdx.x == 0 && checkbits_d != NULL) {
        checkbits_d[blockIdx.x] = 1;
    }
}

}

void *nba::ipv6_route_lookup_get_cuda_kernel() {
    return reinterpret_cast<void *> (ipv6_route_lookup_cuda);
}

// vim: ts=8 sts=4 sw=4 et
