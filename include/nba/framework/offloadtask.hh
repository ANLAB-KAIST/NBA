#ifndef __NBA_OFFLOAD_TASK_HH__
#define __NBA_OFFLOAD_TASK_HH__

#include <nba/core/intrinsic.hh>
#include <nba/core/threading.hh>
#include <nba/core/offloadtypes.hh>
#include <nba/framework/config.hh>
#include <nba/framework/threadcontext.hh>
#include <nba/framework/computecontext.hh>
#include <nba/framework/datablock.hh>
#include <nba/framework/task.hh>
#include <cstdint>
#include <vector>
#include <ev.h>

#define NBA_MAX_OFFLOADED_PACKETS (NBA_MAX_COPROC_PPDEPTH * NBA_MAX_COMPBATCH_SIZE)

namespace nba {

enum TaskStates : int {
    TASK_INITIALIZING = 0,
    TASK_INITIALIZED = 1,
    TASK_PREPARED = 2,
    TASK_H2D_COPYING = 3,
    TASK_EXECUTING = 4,
    TASK_D2H_COPYING = 5,
    TASK_FINISHED = 6
};

/* Forward declarations */
class ElementGraph;
class OffloadableElement;
class PacketBatch;

class OffloadTask {
public:
    OffloadTask();
    virtual ~OffloadTask();

    FixedArray<int, NBA_MAX_DATABLOCKS> datablocks;

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
    struct resource_param res; /* Initialized during execute(). */
    uint64_t offload_start;
    double offload_cost;
    size_t num_pkts;
    size_t num_bytes;
    enum TaskStates state;

    /* Initialized by element graph. */
    struct task_tracker tracker;
    int local_dev_idx;
    struct ev_loop *src_loop;
    comp_thread_context *comp_ctx;
    coproc_thread_context *coproc_ctx;
    ComputeContext *cctx;
    io_base_t io_base;
    uint32_t task_id;
    ElementGraph *elemgraph;
    FixedArray<PacketBatch*, NBA_MAX_COPROC_PPDEPTH> batches;
    FixedArray<int, NBA_MAX_COPROC_PPDEPTH> input_ports;
    OffloadableElement* elem;
    int dbid_h2d[NBA_MAX_DATABLOCKS];

    host_mem_t dbarray_h;
    dev_mem_t dbarray_d;

    struct ev_async *completion_watcher __cache_aligned;
    struct rte_ring *completion_queue __cache_aligned;

private:
    friend class OffloadableElement;

    bool kernel_skipped;

    size_t input_begin;
    size_t inout_begin;
    size_t output_begin;

    size_t last_input_size;   // for debugging
    size_t last_output_size;  // for debugging
};

}

#endif /* __NBA_OFFLOADTASK_HH__ */

// vim: ts=8 sts=4 sw=4 et foldmethod=marker
