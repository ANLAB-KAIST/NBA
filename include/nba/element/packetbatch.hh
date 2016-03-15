#ifndef __NBA_PACKETBATCH_HH__
#define __NBA_PACKETBATCH_HH__
#include <nba/core/intrinsic.hh>
#include <nba/framework/config.hh>
#include <nba/framework/datablock.hh>
#include <nba/framework/task.hh>
#include <nba/element/annotation.hh>
#include <cstdint>
#include <cstring>
#include <vector>
#include <rte_config.h>
#include <rte_memory.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>

extern "C" {
struct rte_ring;
};

/*
 * How to read the macros below:
 * -----------------------------
 *
 * FOR_EACH_PACKET* wraps a per-packet iteration loop over a packet batch.
 * It interally exposes pkt (Packet*) and pkt_idx (unsigned) loop
 * variables.
 *
 * "ALL" modifier means that it does not check packet exclusion,
 * "PREFETCH" modifier means that it performs prefetching with the given depth,
 * and "INIT" modifier means that it is used for batch initialization
 * where the batch is always continuous.
 *
 * Usage:
 * ------
 *
 * Note: these loops cannot be nested!
 *
 * FOR_EACH_PACKET(batch) {
 *   ... batch->packets[pkt_idx] ...
 * } END_FOR;
 *
 * FOR_EACH_PACKET_ALL(batch) {
 *   if (IS_PACKET_VALID(batch, pkt_idx)) ...
 *   ... batch->packets[pkt_idx] ...
 * } END_FOR_ALL;
 *
 * FOR_EACH_PACKET_ALL_PREFETCH(batch, 4u) {
 *   ... batch->packets[pkt_idx] ...
 * } END_FOR_ALL_PREFETCH;
 *
 */

#if NBA_BATCHING_SCHEME == NBA_BATCHING_TRADITIONAL
/* Traditional batching: just skip the excluded packets.
 *
 * This scheme does not reorder packets.
 * The count includes both valid and invalid (excluded) packets.
 */

#define FOR_EACH_PACKET(batch) \
for (unsigned pkt_idx = 0; pkt_idx < batch->count; pkt_idx ++) { \
    if (likely(!batch->excluded[pkt_idx])) {
#define END_FOR \
    } /* endif(!excluded) */ \
} /* endfor(batch) */

#define FOR_EACH_PACKET_ALL(batch) \
{ \
    for (unsigned pkt_idx = 0; pkt_idx < batch->count; pkt_idx ++) {
#define END_FOR_ALL \
    } /* endfor(batch) */ \
}

#define FOR_EACH_PACKET_ALL_PREFETCH(batch, depth) \
{ \
    for (unsigned pkt_idx = 0; pkt_idx < RTE_MIN(depth, batch->count); pkt_idx++) \
        if (batch->packets[pkt_idx] != nullptr) \
            rte_prefetch0(rte_pktmbuf_mtod(batch->packets[pkt_idx], void*)); \
    for (unsigned pkt_idx = 0; pkt_idx < batch->count; pkt_idx ++) { \
        if (pkt_idx + depth < batch->count && batch->packets[pkt_idx + depth] != nullptr) \
            rte_prefetch0(rte_pktmbuf_mtod(batch->packets[pkt_idx + depth], void*));
#define END_FOR_ALL_PREFETCH \
    } /* endfor(batch) */ \
}

#define FOR_EACH_PACKET_ALL_INIT_PREFETCH(batch, depth) \
{ \
    for (unsigned pkt_idx = 0; pkt_idx < RTE_MIN(depth, batch->count); pkt_idx++) { \
        rte_prefetch0(rte_pktmbuf_mtod(batch->packets[pkt_idx], void*)); \
        rte_prefetch0(Packet::from_base_nocheck(batch->packets[pkt_idx])); \
    } \
    for (unsigned pkt_idx = 0; pkt_idx < batch->count; pkt_idx ++) { \
        if (pkt_idx + depth < batch->count) { \
            rte_prefetch0(rte_pktmbuf_mtod(batch->packets[pkt_idx + depth], void*)); \
            rte_prefetch0(Packet::from_base_nocheck(batch->packets[pkt_idx + depth])); \
        }
#define END_FOR_ALL_INIT_PREFETCH \
    } /* endfor(batch) */ \
}

#define INIT_BATCH_MASK(batch) \
{ \
    for (unsigned pkt_idx = 0; pkt_idx < batch->count; pkt_idx++) { \
        batch->excluded[pkt_idx] = false; \
    } \
}
#define IS_PACKET_VALID(batch, pkt_idx) \
    (likely(!batch->excluded[pkt_idx]))
#define IS_PACKET_INVALID(batch, pkt_idx) \
    (unlikely(batch->excluded[pkt_idx]))
#define EXCLUDE_PACKET(batch, pkt_idx) \
{ \
    batch->excluded[pkt_idx] = true; \
    batch->packets[pkt_idx] = nullptr; \
}
#define EXCLUDE_PACKET_MARK_ONLY(batch, pkt_idx) \
    batch->excluded[pkt_idx] = true
#define ADD_PACKET(batch, pkt) \
{ \
    int cnt = batch->count ++; \
    batch->packets[cnt] = pkt; \
    batch->excluded[cnt] = false; \
}


#elif NBA_BATCHING_SCHEME == NBA_BATCHING_CONTINUOUS
/* Continuous batching: sort out excluded packets at the end of batch
 * to completely remove the exclusion check.
 *
 * This scheme may reorder packets.
 * The count includes only valid packets.
 */
#define FOR_EACH_PACKET(batch) \
{ \
    for (unsigned pkt_idx = 0; pkt_idx < batch->count; pkt_idx ++) {
#define END_FOR \
    } /* endfor(batch) */ \
}

#define FOR_EACH_PACKET_ALL(batch) \
{ \
    for (unsigned pkt_idx = 0; pkt_idx < batch->count; pkt_idx ++) {
#define END_FOR_ALL \
    } /* endfor(batch) */ \
}

#define FOR_EACH_PACKET_ALL_PREFETCH(batch, depth) \
{ \
    for (unsigned pkt_idx = 0; pkt_idx < RTE_MIN(depth, batch->count); pkt_idx++) \
        rte_prefetch0(rte_pktmbuf_mtod(batch->packets[pkt_idx], void*)); \
    for (unsigned pkt_idx = 0; pkt_idx < batch->count; pkt_idx ++) { \
        if (pkt_idx + depth < batch->count) \
            rte_prefetch0(rte_pktmbuf_mtod(batch->packets[pkt_idx + depth], void*));
#define END_FOR_ALL_PREFETCH \
    } /* endfor(batch) */ \
}

#define FOR_EACH_PACKET_ALL_INIT_PREFETCH(batch, depth) \
{ \
    for (unsigned pkt_idx = 0; pkt_idx < RTE_MIN(depth, batch->count); pkt_idx++) { \
        rte_prefetch0(rte_pktmbuf_mtod(batch->packets[pkt_idx], void*)); \
        rte_prefetch0(Packet::from_base_nocheck(batch->packets[pkt_idx])); \
    } \
    for (unsigned pkt_idx = 0; pkt_idx < batch->count; pkt_idx ++) { \
        if (pkt_idx + depth < batch->count) { \
            rte_prefetch0(rte_pktmbuf_mtod(batch->packets[pkt_idx + depth], void*)); \
            rte_prefetch0(Packet::from_base_nocheck(batch->packets[pkt_idx + depth])); \
        }
#define END_FOR_ALL_INIT_PREFETCH \
    } /* endfor(batch) */ \
}

#define INIT_BATCH_MASK(batch) \
{ \
    for (unsigned pkt_idx = 0; pkt_idx < batch->count; pkt_idx++) { \
        batch->excluded[pkt_idx] = false; \
    } \
}
#define IS_PACKET_VALID(batch, pkt_idx) true
#define IS_PACKET_INVALID(batch, pkt_idx) false
#define EXCLUDE_PACKET(batch, pkt_idx) \
{ \
    batch->excluded[pkt_idx] = true; \
    batch->packets[pkt_idx] = nullptr; \
    batch->has_dropped = true; \
}
#define EXCLUDE_PACKET_MARK_ONLY(batch, pkt_idx) \
    batch->excluded[pkt_idx] = true
#define ADD_PACKET(batch, pkt) \
{ \
    int cnt = batch->count ++; \
    batch->packets[cnt] = pkt; \
    batch->excluded[cnt] = false; \
}


#elif NBA_BATCHING_SCHEME == NBA_BATCHING_BITVECTOR
/* Bitvector batching: use built-in bit operators over a exclusion mask to
 * efficiently skip the excluded packets without conditional branches.
 *
 * This scheme does not reorder packets.
 * The count includes both valid and invalid (excluded) packets.
 *
 * WARNING: the computation batch size must be <= 64.
 */

#define FOR_EACH_PACKET(batch) \
{ \
    uint64_t _mask = batch->mask; \
    while (_mask != 0) { \
        const unsigned pkt_idx = __builtin_clzll(_mask); \
        _mask &= ~(1llu << (64 - (pkt_idx + 1)));
#define END_FOR \
    } /* endwhile(_mask) */ \
}

#define FOR_EACH_PACKET_ALL(batch) \
{ \
    for (unsigned pkt_idx = 0; pkt_idx < batch->count; pkt_idx ++) {
#define END_FOR_ALL \
    } /* endfor(batch) */ \
}

#define FOR_EACH_PACKET_ALL_PREFETCH(batch, depth) \
{ \
    uint64_t _pmask = batch->mask; \
    uint64_t _mask = batch->mask; \
    unsigned _cnt = 0; \
    while (_pmask != 0 && _cnt < depth) { \
        const unsigned pre_pkt_idx = __builtin_clzll(_pmask); \
        rte_prefetch0(rte_pktmbuf_mtod(batch->packets[pre_pkt_idx], void*)); \
        _pmask &= ~(1llu << (64 - (pre_pkt_idx + 1))); \
        _cnt ++; \
    } \
    while (_mask != 0) { \
        if (_pmask != 0) { \
            const unsigned pre_pkt_idx = __builtin_clzll(_pmask); \
            rte_prefetch0(rte_pktmbuf_mtod(batch->packets[pre_pkt_idx], void*)); \
            _pmask &= ~(1llu << (64 - (pre_pkt_idx + 1))); \
        } \
        const unsigned pkt_idx = __builtin_clzll(_mask); \
        _mask &= ~(1llu << (64 - (pkt_idx + 1)));
#define END_FOR_ALL_PREFETCH \
    } /* endwhile(batch) */ \
}

#define FOR_EACH_PACKET_ALL_INIT_PREFETCH(batch, depth) \
{ \
    assert(batch->count <= 64); \
    for (unsigned pkt_idx = 0; pkt_idx < RTE_MIN(depth, batch->count); pkt_idx++) { \
        rte_prefetch0(rte_pktmbuf_mtod(batch->packets[pkt_idx], void*)); \
        rte_prefetch0(Packet::from_base_nocheck(batch->packets[pkt_idx])); \
    } \
    for (unsigned pkt_idx = 0; pkt_idx < batch->count; pkt_idx ++) { \
        if (pkt_idx + depth < batch->count) { \
            rte_prefetch0(rte_pktmbuf_mtod(batch->packets[pkt_idx + depth], void*)); \
            rte_prefetch0(Packet::from_base_nocheck(batch->packets[pkt_idx + depth])); \
        }
#define END_FOR_ALL_INIT_PREFETCH \
    } /* endfor(batch) */ \
}

#define INIT_BATCH_MASK(batch) \
{ \
    batch->mask = (0xFFFFffffFFFFffffllu << (64 - batch->count)); \
}
#define IS_PACKET_VALID(batch, pkt_idx) \
    (likely((batch->mask & (1llu << (64 - (pkt_idx + 1)))) == 1))
#define IS_PACKET_INVALID(batch, pkt_idx) \
    (unlikely((batch->mask & (1llu << (64 - (pkt_idx + 1)))) == 0))
#define EXCLUDE_PACKET(batch, pkt_idx) \
{ \
    batch->mask &= ~(1llu << (64 - (pkt_idx + 1))); \
    batch->packets[pkt_idx] = nullptr; \
}
#define EXCLUDE_PACKET_MARK_ONLY(batch, pkt_idx) \
{ \
    batch->mask &= ~(1llu << (64 - (pkt_idx + 1))); \
}
#define ADD_PACKET(batch, pkt) \
{ \
    int cnt = batch->count ++; \
    batch->packets[cnt] = pkt; \
    batch->mask |= (1llu << (64 - (pkt_idx + 1))); \
}


#elif NBA_BATCHING_SCHEME == NBA_BATCHING_LINKEDLIST
/* Linked-list batching: batches are no longer arrays.
 * We store "next packet" pointers inside the packets and use them to
 * retrieve the next packet in the batch and stop when it is nullptr.
 */

#define FOR_EACH_PACKET(batch) \
{ \
    int _next_idx = batch->first_idx; \
    while (_next_idx != -1) { \
        const unsigned pkt_idx = (unsigned) _next_idx;
#define END_FOR \
        /*printf("pkt = %p\n", batch->packets[pkt_idx]);*/ \
        _next_idx = Packet::from_base(batch->packets[pkt_idx])->next_idx; \
    } /* endwhile(batch) */ \
}

#define FOR_EACH_PACKET_ALL(batch) \
{ \
    int _next_idx = batch->first_idx; \
    while (_next_idx != -1) { \
        const unsigned pkt_idx = (unsigned) _next_idx;
#define END_FOR_ALL \
        _next_idx = Packet::from_base(batch->packets[pkt_idx])->next_idx; \
    } /* endwhile(batch) */ \
}

// TODO: apply prefetching??
#define FOR_EACH_PACKET_ALL_PREFETCH(batch, depth) \
{ \
    int _next_idx = batch->first_idx; \
    while (_next_idx != -1) { \
        const unsigned pkt_idx = (unsigned) _next_idx;
#define END_FOR_ALL_PREFETCH \
        _next_idx = Packet::from_base(batch->packets[pkt_idx])->next_idx; \
    } /* endwhile(batch) */ \
}

/* Initialization is a special case where we need to manually set up
 * next/prev pointers of all packets. */
#define FOR_EACH_PACKET_ALL_INIT_PREFETCH(batch, depth) \
{ \
    for (unsigned pkt_idx = 0; pkt_idx < RTE_MIN(depth, batch->count); pkt_idx++) { \
        rte_prefetch0(rte_pktmbuf_mtod(batch->packets[pkt_idx], void*)); \
        rte_prefetch0(Packet::from_base_nocheck(batch->packets[pkt_idx])); \
    } \
    for (unsigned pkt_idx = 0; pkt_idx < batch->count; pkt_idx ++) { \
        if (pkt_idx + depth < batch->count) { \
            rte_prefetch0(rte_pktmbuf_mtod(batch->packets[pkt_idx + depth], void*)); \
            rte_prefetch0(Packet::from_base_nocheck(batch->packets[pkt_idx + depth])); \
        }
#define END_FOR_ALL_INIT_PREFETCH \
    } /* endfor(batch) */ \
}

#define INIT_BATCH_MASK(batch) \
{ \
    /* do nothing. */ \
}
#define IS_PACKET_VALID(batch, pkt_idx) true
#define IS_PACKET_INVALID(batch, pkt_idx) false
#define EXCLUDE_PACKET(batch, pkt_idx) \
{ \
    Packet *pkt = Packet::from_base(batch->packets[pkt_idx]); \
    int p = pkt->prev_idx; \
    int n = pkt->next_idx; \
    if (p != -1) \
        Packet::from_base(batch->packets[p])->next_idx = n; \
    else \
        batch->first_idx = n; \
    if (n != -1) \
        Packet::from_base(batch->packets[n])->prev_idx = p; \
    else \
        batch->last_idx = p; \
    /*batch->packets[pkt_idx] = nullptr;*/ \
    batch->count --; \
}
#define EXCLUDE_PACKET_MARK_ONLY(batch, pkt_idx) \
{ \
    /* do nothing. */ \
}
#define ADD_PACKET(batch, raw_pkt) \
{ \
    int l = batch->last_idx; \
    Packet *pkt = Packet::from_base(raw_pkt); \
    assert(batch->slot_count < NBA_MAX_COMP_BATCH_SIZE); \
    unsigned new_idx = batch->slot_count ++; \
    if (l != -1) { \
        pkt->prev_idx = l; \
        pkt->next_idx = -1; \
        Packet::from_base(batch->packets[l])->next_idx = new_idx; \
    } else { \
        pkt->prev_idx = -1; \
        pkt->next_idx = -1; \
        batch->first_idx = new_idx; \
    } \
    batch->packets[new_idx] = raw_pkt; \
    batch->last_idx = new_idx; \
    batch->count ++; \
}


#else
    #error Unknown NBA_BATCHING_SCHEME
#endif


namespace nba {

class Element;

enum BatchDisposition {
    KEPT_BY_ELEMENT = -1,
    CONTINUE_TO_PROCESS = 0,
};

class PacketBatch {
public:
    PacketBatch()
        : count(0),
          #if NBA_BATCHING_SCHEME == NBA_BATCHING_CONTINUOUS
          drop_count(0),
          #endif
          #if NBA_BATCHING_SCHEME == NBA_BATCHING_BITVECTOR
          mask(0),
          #endif
          #if NBA_BATCHING_SCHEME == NBA_BATCHING_LINKEDLIST
          first_idx(-1), last_idx(-1), slot_count(0),
          #endif
          datablock_states(nullptr), recv_timestamp(0),
          generation(0), batch_id(0),
          #if NBA_BATCHING_SCHEME == NBA_BATCHING_CONTINUOUS
          has_dropped(false),
          #endif
          delay_start(0), compute_time(0)
    {
        #ifdef DEBUG
        memset(&results[0], 0xdd, sizeof(int) * NBA_MAX_COMP_BATCH_SIZE);
        #if (NBA_BATCHING_SCHEME == NBA_BATCHING_TRADITIONAL) \
            || (NBA_BATCHING_SCHEME == NBA_BATCHING_CONTINUOUS)
        memset(&excluded[0], 0xcc, sizeof(bool) * NBA_MAX_COMP_BATCH_SIZE);
        #endif
        memset(&packets[0], 0xbb, sizeof(struct rte_mbuf*) * NBA_MAX_COMP_BATCH_SIZE);
        #endif
    }

    virtual ~PacketBatch()
    {
    }

    #if NBA_BATCHING_SCHEME == NBA_BATCHING_CONTINUOUS
    /**
     * Moves excluded packets to the end of batches, by swapping them
     * with the tail packets, to reduce branching overheads when iterating
     * over the packet batch in many places.
     * (We assume that this "in-batch" reordering does not incur performance
     * overheads for transport layers.)
     * It stores the number of dropped packets to drop_count member
     * variable.  Later, ElementGraph refer this value to actually free
     * the excluded packets.
     *
     * This should only be called right after doing Element::_process_batch()
     * or moving packets to other batches in ElementGraph.
     * This may be called multiple times until reaching the next element.
     */
    void collect_excluded_packets();

    /**
     * Moves the collected excluded packets at the tail to drop_queue,
     * and resets drop_count to zero.
     */
    void clean_drops(struct rte_ring *drop_queue);
    #endif

    unsigned count;
    #if NBA_BATCHING_SCHEME == NBA_BATCHING_CONTINUOUS
    unsigned drop_count;
    #endif
    #if NBA_BATCHING_SCHEME == NBA_BATCHING_BITVECTOR
    uint64_t mask;
    #endif
    #if NBA_BATCHING_SCHEME == NBA_BATCHING_LINKEDLIST
    int first_idx;
    int last_idx;
    unsigned slot_count;
    #endif
    struct datablock_tracker *datablock_states;
    uint64_t recv_timestamp;
    uint64_t generation;
    uint64_t batch_id;
    struct task_tracker tracker;
    #if NBA_BATCHING_SCHEME == NBA_BATCHING_CONTINUOUS
    bool has_dropped;
    #endif
    uint64_t delay_start;
    uint64_t delay_time;
    double compute_time;

    struct annotation_set banno __cache_aligned;  /** Batch-level annotations. */
    #if (NBA_BATCHING_SCHEME == NBA_BATCHING_TRADITIONAL) \
        || (NBA_BATCHING_SCHEME == NBA_BATCHING_CONTINUOUS)
    bool excluded[NBA_MAX_COMP_BATCH_SIZE] __cache_aligned;
    #endif
    struct rte_mbuf *packets[NBA_MAX_COMP_BATCH_SIZE] __cache_aligned;
    int results[NBA_MAX_COMP_BATCH_SIZE] __cache_aligned;
};

}

#endif

// vim: ts=8 sts=4 sw=4 et
