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

#define NBA_MAX_OFFLOADED_PACKETS (NBA_MAX_COPROC_PPDEPTH * NBA_MAX_COMPBATCH_SIZE)

namespace nba {

typedef std::function<void(ComputeContext *ctx, struct resource_param *res, struct annotation_set **anno_ptr_array)> offload_compute_handler;
typedef std::function<void(ComputeDevice *dev)> offload_init_handler;

class ElementGraph;
class OffloadableElement;

class OffloadTask {
public:
    OffloadTask();
    virtual ~OffloadTask();

    /* Executed in computation threads. */
    size_t calculate_buffer_sizes();  // returns the number of valid packets
    void prepare_host_buffer();
    void postproc();

    /* Executed in coprocessor threads.
     * Copies prepared IO buffers to the device, and calls the kernel
     * launch handler provided by the offloadable element. */
    void copy_buffers_h2d();
    void execute();
    void copy_buffers_d2h();

    bool poll_kernel_finished();
    bool poll_d2h_copy_finished();

    /* Executed in CUDA's own threads or coprocessor threads
     * depending on the compute context implementation. */
    void notify_completion();

public:
    /* Initialized during execute(). */
    void    *input_buffer_h;
    memory_t input_buffer_d;
    size_t  *aligned_elemsizes_h;
    memory_t aligned_elemsizes_d;
    size_t  *input_elemsizes_h;
    memory_t input_elemsizes_d;
    size_t   input_buffer_size;

    void    *output_buffer_h;
    memory_t output_buffer_d;
    size_t  *output_elemsizes_h;
    memory_t output_elemsizes_d;
    size_t output_buffer_size;

    struct resource_param res;

    uint64_t begin_timestamp;
    uint64_t offload_start;
    double offload_cost;
    int local_dev_idx;

    /* Initialized by element graph. */
    struct ev_loop *src_loop;
    coproc_thread_context *coproc_ctx;
    ComputeDevice *device;
    ComputeContext *cctx;
    ElementGraph *elemgraph;
    offload_compute_handler handler;
    size_t num_batches;
    PacketBatch* batches[NBA_MAX_COPROC_PPDEPTH];
    int input_ports[NBA_MAX_COPROC_PPDEPTH];
    /* Currently only one offloaded element is present.
     * If we implement slicing optimization later,
     * this may have multiple subsequent offloaded elements. */
    std::vector<OffloadableElement*> offloaded_elements;

    struct ev_async *completion_watcher __rte_cache_aligned;
    struct rte_ring *completion_queue __rte_cache_aligned;

    /* Store pointers of each offloaded packet's annoation_set.
     * In OffloadableElement::preproc(), user can store some values into 
     * annotation of proprocessed packet.
     * Those valses are passed to cuda_compute_handler() 
     * in the form of anntation_set and then can be used 
     * as a parameter of offloaded computation. */
    struct annotation_set *anno_ptr_array[NBA_MAX_OFFLOADED_PACKETS];

private:
    struct input_roi_info input_roi;
    struct output_roi_info output_roi;
    size_t total_num_pkts;

    static unsigned relative_offsets[3];
};

}

#endif

// vim: ts=8 sts=4 sw=4 et foldmethod=marker
