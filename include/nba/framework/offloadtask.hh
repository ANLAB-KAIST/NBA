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

enum TaskStates {
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
    struct resource_param res;
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
    ElementGraph *elemgraph;
    FixedArray<PacketBatch*, nullptr, NBA_MAX_COPROC_PPDEPTH> batches;
    FixedArray<int, -1, NBA_MAX_COPROC_PPDEPTH> input_ports;
    OffloadableElement* elem;
    int dbid_h2d[NBA_MAX_DATABLOCKS];

    struct datablock_kernel_arg *dbarray_h;
    memory_t dbarray_d;

    struct ev_async *completion_watcher __cache_aligned;
    struct rte_ring *completion_queue __cache_aligned;

    uint64_t task_id; // for deubgging
private:
    friend class OffloadableElement;

    void *host_write_begin;
    void *host_read_begin;
    memory_t dev_write_begin;
    memory_t dev_read_begin;
    size_t input_alloc_size_begin;
    size_t output_alloc_size_begin;

    size_t last_input_size;
    size_t last_output_size;
};

}

#endif /* __NBA_OFFLOADTASK_HH__ */

// vim: ts=8 sts=4 sw=4 et foldmethod=marker
