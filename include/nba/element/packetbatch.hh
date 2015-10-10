#ifndef __NBA_PACKETBATCH_HH__
#define __NBA_PACKETBATCH_HH__
#include <nba/core/intrinsic.hh>
#include <nba/framework/config.hh>
#include <nba/framework/datablock.hh>
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
}

namespace nba {

class Element;
#define NBA_BATCHING_TRADITIONAL

#ifdef NBA_BATCHING_TRADITIONAL
/* Traditional batching: just skip the excluded packets. */
/*
 * FOR_EACH_PACKET wraps a per-packet iteration loop over a packet batch.
 * It interally exposes pkt (Packet*) and pkt_idx (unsigned) loop
 * variables.
 */
#define FOR_EACH_PACKET(batch) \
    for (unsigned pkt_idx = 0; pkt_idx < batch->count; pkt_idx ++) { \
        if (likely(!batch->excluded[pkt_idx])) {
#define END_FOR \
        } /* endif(!excluded) */ \
} /* endfor(batch) */
#endif

#ifdef NBA_BATCHING_CONTINUOUS
/* Continuous batching: sort out excluded packets at the end of batch
 * to completely remove the exclusion check. */
// TODO: add ifdef checks to batch reorganization codes.
#define FOR_EACH_PACKET(batch) \
    for (unsigned pkt_idx = 0; pkt_idx < batch->count; pkt_idx ++) { \
#define END_FOR \
} /* endfor(batch) */
#endif

#ifdef NBA_BATCHING_BITVECTOR
/* Bitvector batching: use built-in bit operators over a exclusion mask to
 * efficiently skip the excluded packets without conditional branches. */
/* WARNING: the computation batch size must be <= 64. */
// TODO: implement
#define FOR_EACH_PACKET(batch) { \
    for (unsigned pkt_idx = 0; pkt_idx < batch->count; pkt_idx ++) { \
        if (likely(!batch->excluded[pkt_idx])) {
#define END_FOR } /* endif(!excluded) */ \
    } /* endfor(batch) */ \
}
#endif

#ifdef NBA_BATCHING_LINKEDLIST
/* Linked-list batching: batches are no longer arrays.
 * We store "next packet" pointers inside the packets and use them to
 * retrieve the next packet in the batch and stop when it is nullptr.
 */
// TODO: backport from the linked-list-batch branch.
    #error NBA_BATCHING_LINKEDLIST is not implemented yet.
#endif

enum BatchDisposition {
    KEPT_BY_ELEMENT = -1,
    CONTINUE_TO_PROCESS = 0,
};

class PacketBatch {
public:
    PacketBatch()
        : count(0), drop_count(0), mask(0), datablock_states(nullptr), recv_timestamp(0),
          generation(0), batch_id(0), element(nullptr), input_port(0), has_results(false),
          has_dropped(false), delay_start(0), compute_time(0)
    {
        #ifdef DEBUG
        memset(&results[0], 0xdd, sizeof(int) * NBA_MAX_COMP_BATCH_SIZE);
        memset(&excluded[0], 0xcc, sizeof(bool) * NBA_MAX_COMP_BATCH_SIZE);
        memset(&packets[0], 0xbb, sizeof(struct rte_mbuf*) * NBA_MAX_COMP_BATCH_SIZE);
        #endif
    }

    virtual ~PacketBatch()
    {
    }

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

    unsigned count;
    unsigned drop_count;
    uint64_t mask;
    struct datablock_tracker *datablock_states;
    uint64_t recv_timestamp;
    uint64_t generation;
    uint64_t batch_id;
    Element* element;
    int input_port;
    bool has_results;
    bool has_dropped;
    uint64_t delay_start;
    uint64_t delay_time;
    double compute_time;

    struct annotation_set banno __rte_cache_aligned;  /** Batch-level annotations. */
    bool excluded[NBA_MAX_COMP_BATCH_SIZE] __rte_cache_aligned;
    struct rte_mbuf *packets[NBA_MAX_COMP_BATCH_SIZE] __rte_cache_aligned;
    int results[NBA_MAX_COMP_BATCH_SIZE] __rte_cache_aligned;
};

}

#endif

// vim: ts=8 sts=4 sw=4 et
