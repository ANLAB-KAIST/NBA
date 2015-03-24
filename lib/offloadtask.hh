#ifndef __NBA_OFFLOAD_TASK_HH__
#define __NBA_OFFLOAD_TASK_HH__

#include <cstdint>
#include <functional>
#include <vector>
#include "config.hh"
#include "types.hh"
#include "thread.hh"
#include "computecontext.hh"
#include "annotation.hh"
#include "packetbatch.hh"
#include "datablock.hh"

#define NBA_MAX_OFFLOADED_PACKETS (NBA_MAX_COPROC_PPDEPTH * NBA_MAX_COMPBATCH_SIZE)

namespace nba {

typedef std::function<void(ComputeContext *ctx, struct resource_param *res)> offload_compute_handler;
typedef std::function<void(ComputeDevice *dev)> offload_init_handler;

class ElementGraph;
class OffloadableElement;

class OffloadTask {
public:
    OffloadTask();
    virtual ~OffloadTask();

    FixedArray<int, -1, NBA_MAX_DATABLOCKS> datablocks;

    /* Executed in worker threads. */
    void prepare_read_buffer();
    void prepare_write_buffer();
    void postprocess();

    /* Executed in coprocessor threads.
     * Copies prepared IO buffers to the device, and calls the kernel
     * launch handler provided by the offloadable element. */
    bool copy_h2d();
    void execute();
    bool copy_d2h();
    bool poll_kernel_finished();
    bool poll_d2h_copy_finished();

    /* Executed in CUDA's own threads or coprocessor threads
     * depending on the compute context implementation. */
    void notify_completion();

public:
    /* Initialized during execute(). */
    uint64_t begin_timestamp;
    struct resource_param res;
    uint64_t offload_start;
    double offload_cost;

    /* Initialized by element graph. */
    int local_dev_idx;
    struct ev_loop *src_loop;
    comp_thread_context *comp_ctx;
    coproc_thread_context *coproc_ctx;
    ComputeContext *cctx;
    ElementGraph *elemgraph;
    FixedArray<PacketBatch*, nullptr, NBA_MAX_COPROC_PPDEPTH> batches;
    FixedArray<int, -1, NBA_MAX_COPROC_PPDEPTH> input_ports;
    OffloadableElement* elem;
    int dbid_h2d[NBA_MAX_DATABLOCKS];

    struct datablock_kernel_arg *dbarray_h;
    memory_t dbarray_d;

    struct ev_async *completion_watcher __rte_cache_aligned;
    struct rte_ring *completion_queue __rte_cache_aligned;
};

}

#endif

// vim: ts=8 sts=4 sw=4 et foldmethod=marker
