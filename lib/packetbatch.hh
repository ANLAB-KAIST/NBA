#ifndef __NBA_PACKETBATCH_HH__
#define __NBA_PACKETBATCH_HH__
#include "types.hh"
#include <cstdint>
#include <cstring>
extern "C" {
#include <rte_config.h>
#include <rte_memory.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
}
#include "config.hh"
#include "annotation.hh"

namespace nba {

enum BatchDisposition {
    KEPT_BY_ELEMENT = -1,
    CONTINUE_TO_PROCESS = 0,
};

class PacketBatch {
public:
    PacketBatch()
        : count(0), offloaded_device(-1), recv_timestamp(0), generation(0), has_results(false),
          delay_start(0)
,compute_time(0)
    {
        #ifdef DEBUG
        memset(&results[0], 0xdd, sizeof(int) * NBA_MAX_COMPBATCH_SIZE);
        memset(&excluded[0], 0xcc, sizeof(bool) * NBA_MAX_COMPBATCH_SIZE);
        memset(&packets[0], 0xbb, sizeof(struct rte_mbuf*) * NBA_MAX_COMPBATCH_SIZE);
        memset(&annos[0], 0xaa, sizeof(struct annotation_set) * NBA_MAX_COMPBATCH_SIZE);
        #endif
    }

    virtual ~PacketBatch()
    {
    }

    unsigned count;
    int offloaded_device;
    uint64_t recv_timestamp;
    uint64_t generation;
    bool has_results;
    uint64_t delay_start;

    uint64_t batch_id;
    Element* element;
    int input_port;
    struct annotation_set banno __rte_cache_aligned;

    bool excluded[NBA_MAX_COMPBATCH_SIZE] __rte_cache_aligned;
    struct rte_mbuf *packets[NBA_MAX_COMPBATCH_SIZE] __rte_cache_aligned;
    struct annotation_set annos[NBA_MAX_COMPBATCH_SIZE] __rte_cache_aligned;
    int results[NBA_MAX_COMPBATCH_SIZE] __rte_cache_aligned;
    double compute_time;
};

}

#endif

// vim: ts=8 sts=4 sw=4 et
