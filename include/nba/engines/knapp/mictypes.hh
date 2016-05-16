#ifndef __NBA_KNAPP_MICTYPES_HH__
#define __NBA_KNAPP_MICTYPES_HH__

#ifndef __MIC__
#error "This header should be used by MIC-side codes only."
#endif
#ifdef __NBA_KNAPP_HOSTTYPES_HH__
#error "Mixed use of MIC/host headers!"
#endif

#include <cstdint>
#include <atomic>
#include <vector>
#include <functional>
#include <scif.h>
#include <nba/engines/knapp/defs.hh>
#include <nba/engines/knapp/sharedtypes.hh>
#include <nba/engines/knapp/micintrinsic.hh>
#include <nba/engines/knapp/micbarrier.hh>

namespace nba { namespace knapp {

/* Forward decls. */
struct work;
class PollRing;
class RMABuffer;

/* Callback defs. */
typedef std::function<void(
        uint32_t begin_idx,
        uint32_t end_idx,
        struct datablock_kernel_arg **datablocks,
        uint32_t *item_counts,
        uint32_t num_batches,
        size_t num_args,
        void **args)> worker_func_t;


/* MIC-side vDevice context */
struct vdevice {
    int device_id;
    uint32_t pipeline_depth;
    uint32_t ht_per_core;

    scif_epd_t data_epd;
    scif_epd_t ctrl_epd;

    scif_epd_t master_epd;
    scif_epd_t data_listen_epd;
    scif_epd_t ctrl_listen_epd;
    struct scif_portID ctrl_port;
    struct scif_portID mic_data_port;

    PollRing *poll_rings[KNAPP_VDEV_MAX_POLLRINGS];
    RMABuffer *rma_buffers[KNAPP_VDEV_MAX_RMABUFFERS];
    RMABuffer *task_params;
    RMABuffer *d2h_params;

    pthread_barrier_t *master_ready_barrier;
    pthread_barrier_t *term_barrier;
    Barrier **data_ready_barriers;
    Barrier **task_done_barriers;

    pthread_t master_thread;
    pthread_t *worker_threads;
    bool threads_alive;

    struct worker_thread_info *thread_info_array;
    unsigned master_core; //former master_cpu
    unsigned num_worker_threads;
    unsigned next_task_id;
    unsigned cur_task_id;
    //unsigned num_packets_in_cur_task;

    struct work **per_thread_work_info;
    worker_func_t worker_func;

    uint32_t offload_batch_size;
    std::vector<int> pcores;
    std::vector<int> lcores;

    void *_reserved0[0] __cache_aligned;

    std::atomic<bool> exit __cache_aligned;

    void *_reserved1[0] __cache_aligned;

    /* stats-related */
    bool first_entry;
    uint64_t ts_laststat;
    uint64_t ts_curstat;
    uint64_t ts_batch_begin;
    uint64_t ts_batch_end;
    uint64_t total_packets_processed;
    uint64_t total_batches_processed;
    uint64_t acc_batch_process_us;
    uint64_t acc_batch_process_us_sq;
    uint64_t acc_batch_transfer_us;
    uint64_t acc_batch_transfer_us_sq;
};

struct work {
    // Initialized on start-up.
    int thread_id;
    struct vdevice *vdev;
    Barrier *data_ready_barrier;
    Barrier *task_done_barrier;

    // Updated for each offloaded task.
    uint32_t begin_idx;
    uint32_t num_items;
    uint32_t num_args;
    uint64_t kernel_id;
    void *args[KNAPP_MAX_KERNEL_ARGS] __cache_aligned;

    // For termination.
    std::atomic<bool> exit;
} __cache_aligned;

struct worker_thread_info {
     int thread_id;
     struct vdevice *vdev;
    pthread_barrier_t *worker_ready_barrier;
} __cache_aligned;

}} // endns(nba::knapp)

#endif // __NBA_KNAPP_MICTYPES_HH__

// vim: ts=8 sts=4 sw=4 et
