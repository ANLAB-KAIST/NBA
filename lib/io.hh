#ifndef __IO_HH__
#define __IO_HH__

#include "types.hh"
#include "thread.hh"
#include "annotation.hh"

#include <rte_memory.h>
#include <rte_atomic.h>


namespace nshader {

struct io_port_stat {
    uint64_t num_recv_pkts;
    uint64_t num_sent_pkts;
    uint64_t num_sw_drop_pkts;
    uint64_t num_rx_drop_pkts;
    uint64_t num_tx_drop_pkts;
    uint64_t num_invalid_pkts;
    uint64_t num_recv_bytes;
    uint64_t num_sent_bytes;
} __rte_cache_aligned;

struct io_port_stat_atomic {
    rte_atomic64_t num_recv_pkts;
    rte_atomic64_t num_sent_pkts;
    rte_atomic64_t num_sw_drop_pkts;
    rte_atomic64_t num_rx_drop_pkts;
    rte_atomic64_t num_tx_drop_pkts;
    rte_atomic64_t num_invalid_pkts;
    rte_atomic64_t num_recv_bytes;
    rte_atomic64_t num_sent_bytes;
} __rte_cache_aligned;

struct io_thread_stat {
    unsigned num_ports;
    struct io_port_stat port_stats[NSHADER_MAX_PORTS];
} __rte_cache_aligned;

struct io_node_stat {
    unsigned node_id;
    uint64_t last_time;
    struct io_thread_stat last_total;
    unsigned num_threads;
    unsigned num_ports;
    struct io_port_stat_atomic port_stats[NSHADER_MAX_PORTS];
} __rte_cache_aligned;

void io_tx_batch(struct io_thread_context *ctx, PacketBatch *batch);
//void *io_loop(void *arg);
int io_loop(void *arg);

}

#endif

// vim: ts=8 sts=4 sw=4 et
