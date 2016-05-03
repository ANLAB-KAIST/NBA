#ifndef __NBA_KNAPP_SHAREDTYPES_HH__
#define __NBA_KNAPP_SHAREDTYPES_HH__

#include <cstdint>
#include <scif.h>
#ifdef __MIC__
#include <nba/engines/knapp/micintrinsic.hh>
#else
#include <nba/core/intrinsic.hh>
#include <rte_ring.h>
#endif

namespace nba { namespace knapp {

typedef enum : uint8_t {
    OP_SET_WORKLOAD_TYPE, // Followed by workload type identifier (4B)
    OP_MALLOC,            // Followed by buffer size (8B)
    OP_REG_DATA,          // Followed by data offset to be registered(8B)
    OP_REG_POLLRING,      // Followed by number of rings (4B) and poll-ring base offset (8B)
    OP_SEND_DATA,         // Followed by data size (8B) and scif_write to data channel
    NUM_OFFLOAD_OPS
} ctrl_msg_t;

typedef enum : uint8_t {
    RESP_SUCCESS = 0,
    RESP_FAILED = 1
} ctrl_resp_t;

struct taskitem {
    int32_t task_id;      // doubles as poll/buffer index
    uint64_t input_size;
    int32_t num_packets;
} __cache_aligned;

struct bufarray {
    uint8_t **bufs;
    off_t *ra_array;
    uint32_t size;
    uint64_t elem_size;
    uint64_t elem_alloc_size;
    bool uses_ra;
    bool initialized;
} __cache_aligned;

struct poll_ring {
    uint64_t volatile *ring;
    int32_t alloc_bytes;
    off_t ring_ra;
    uint32_t len;
    uint32_t count;
#ifndef __MIC__
    struct rte_ring *id_pool; // Assumes sequential ordering
#endif
} __cache_aligned;


}} // endns(nba::knapp)

#endif // __NBA_KNAPP_SHAREDTYPES_HH__

// vim: ts=8 sts=4 sw=4 et
