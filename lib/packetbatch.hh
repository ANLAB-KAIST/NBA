#ifndef __NBA_PACKETBATCH_HH__
#define __NBA_PACKETBATCH_HH__
#include "types.hh"
#include <cstdint>
#include <cstring>
#include <vector>
#include <rte_config.h>
#include <rte_memory.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>

#include "config.hh"
#include "annotation.hh"
#include "datablock.hh"

namespace nba {

enum BatchDisposition {
    KEPT_BY_ELEMENT = -1,
    CONTINUE_TO_PROCESS = 0,
};

class Packet;

class PacketBatch {
public:
    PacketBatch()
        : count(0), first_packet(nullptr), datablock_states(nullptr), recv_timestamp(0),
          generation(0), batch_id(0), element(nullptr), input_port(0), has_results(false),
          delay_start(0), compute_time(0)
    { }

    virtual ~PacketBatch()
    { }

    unsigned count;
    Packet *first_packet;
    struct datablock_tracker *datablock_states;
    uint64_t recv_timestamp;
    uint64_t generation;
    uint64_t batch_id;
    Element* element;
    int input_port;
    bool has_results;
    uint64_t delay_start;
    double compute_time;
    struct annotation_set banno __rte_cache_aligned;  /** Batch-level annotations. */
};

}

#endif

// vim: ts=8 sts=4 sw=4 et
