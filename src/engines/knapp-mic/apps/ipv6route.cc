#include <nba/core/accumidx.hh>
#include <nba/engines/knapp/defs.hh>
#include <nba/engines/knapp/mictypes.hh>
#include <nba/engines/knapp/sharedtypes.hh>
#include <nba/engines/knapp/kernels.hh>
#include "ipv6route.hh"
#include <cstdio>
#include <algorithm>
#include <utility>

namespace nba { namespace knapp {

extern worker_func_t worker_funcs[];

static void ipv6_route_lookup(
        uint32_t begin_idx,
        uint32_t end_idx,
        struct datablock_kernel_arg **datablocks,
        uint32_t *item_counts,
        uint32_t num_batches,
        size_t num_args,
        void **args);

typedef union {
    uint8_t u8[16];
    uint16_t u16[8];
    uint32_t u32[4];
    uint64_t u64[2];
} __attribute__((packed)) uint128_t;

struct Item {
    uint128_t key;
    uint16_t val;
    uint16_t state;
    uint32_t next;
};
#define IPV6_DEFAULT_HASHTABLE_SIZE 65536
#define IPV6_HASHTABLE_MARKER 0x0002
#define IPV6_HASHTABLE_EMPTY  0x0000
#define IPV6_HASHTABLE_PREFIX 0x0001

static inline uint64_t ntohll(uint64_t val)
{
    return ( (((val) >> 56) & 0x00000000000000ff) | (((val) >> 40) & 0x000000000000ff00) | \
            (((val) >> 24) & 0x0000000000ff0000) | (((val) >>  8) & 0x00000000ff000000) | \
            (((val) <<  8) & 0x000000ff00000000) | (((val) << 24) & 0x0000ff0000000000) | \
            (((val) << 40) & 0x00ff000000000000) | (((val) << 56) & 0xff00000000000000) );
}

#define jhash_mix(a, b, c) \
{ \
  a -= b; a -= c; a ^= (c>>13); \
  b -= c; b -= a; b ^= (a<<8); \
  c -= a; c -= b; c ^= (b>>13); \
  a -= b; a -= c; a ^= (c>>12);  \
  b -= c; b -= a; b ^= (a<<16); \
  c -= a; c -= b; c ^= (b>>5); \
  a -= b; a -= c; a ^= (c>>3);  \
  b -= c; b -= a; b ^= (a<<10); \
  c -= a; c -= b; c ^= (b>>15); \
}

/* The golden ration: an arbitrary value */
#define JHASH_GOLDEN_RATIO  0x9e3779b9

static uint32_t gpu_jhash2_optimized(uint32_t k0, uint32_t k1, uint32_t k2, uint32_t k3)
{
    uint32_t a, b, c;

    a = b = JHASH_GOLDEN_RATIO;
    c = 0;

    a += k0;
    b += k1;
    c += k2;
    jhash_mix(a, b, c);

    c += 4 * 4;
    a += k3;
    jhash_mix(a, b, c);

    return c;
}

static uint32_t hashtable_find(
        uint64_t ip0, uint64_t ip1,
        const Item* table, const int tablesize)
{
    uint32_t index = gpu_jhash2_optimized(
            ip0 >> 32, (uint32_t)ip0,
            ip1 >> 32, (uint32_t)ip1)
            % tablesize;

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

static uint32_t gpu_route_lookup_one(
        uint64_t ip0, uint64_t ip1,
        Item* tables[], size_t* table_sizes)
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

}} //endns(nba::knapp)

using namespace nba::knapp;

static void nba::knapp::ipv6_route_lookup (
        uint32_t begin_idx,
        uint32_t end_idx,
        struct datablock_kernel_arg **datablocks,
        uint32_t *item_counts,
        uint32_t num_batches,
        size_t num_args,
        void **args)
{
    struct datablock_kernel_arg *db_daddrs  = datablocks[0];
    struct datablock_kernel_arg *db_results = datablocks[1];
    Item       **tables_d = static_cast<Item **>(args[0]);
    size_t *table_sizes_d = static_cast<size_t *>(args[1]);

    uint32_t batch_idx, item_idx;
    nba::get_accum_idx(item_counts, num_batches,
                       begin_idx, batch_idx, item_idx);

    for (uint32_t idx = 0; idx < end_idx - begin_idx; ++idx) {
        if (item_idx == item_counts[batch_idx]) {
            batch_idx ++;
            item_idx = 0;
        }

        void *dbase = db_daddrs->batches[batch_idx].buffer_bases;
        void *rbase = db_results->batches[batch_idx].buffer_bases;
        uint128_t daddr
                = (static_cast<uint128_t*>(dbase))[item_idx];
        uint16_t &lookup_result
                = (static_cast<uint16_t*>(rbase))[item_idx];

        lookup_result = 0xffff;
        std::swap(daddr.u64[0], daddr.u64[1]);
        daddr.u64[1] = ntohll(daddr.u64[0]);
        daddr.u64[0] = ntohll(daddr.u64[1]);
        lookup_result = gpu_route_lookup_one(daddr.u64[0], daddr.u64[1],
                                             tables_d, table_sizes_d);

        item_idx ++;
    }
}


void __attribute__((constructor, used)) ipv6_route_lookup_register()
{
    worker_funcs[ID_KERNEL_IPV6LOOKUP] = ipv6_route_lookup;
}


// vim: ts=8 sts=4 sw=4 et
