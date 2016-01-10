/**
 * NBA's Coprocessor Handler.
 *
 * Author: Joongi Kim <joongi@an.kaist.ac.kr>
 */

#include <nba/core/intrinsic.hh>
#include <nba/core/threading.hh>
#include <nba/core/queue.hh>
#include <nba/element/packetbatch.hh>
#include <nba/framework/threadcontext.hh>
#include <nba/framework/logging.hh>
#include <nba/framework/computation.hh>
#include <nba/framework/coprocessor.hh>
#include <nba/framework/offloadtask.hh>
#include <nba/framework/computedevice.hh>
#ifdef USE_CUDA
#include <nba/engines/cuda/computedevice.hh>
#endif
#ifdef USE_PHI
#include <nba/engines/phi/computedevice.hh>
#endif
#include <nba/engines/dummy/computedevice.hh>

#include <unistd.h>
#include <numa.h>
#include <sys/prctl.h>
#include <rte_config.h>
#include <rte_common.h>
#include <rte_cycles.h>
#include <rte_malloc.h>
#include <rte_per_lcore.h>
#include <ev.h>
#ifdef USE_NVPROF
#include <nvToolsExt.h>
#endif

RTE_DECLARE_PER_LCORE(unsigned, _socket_id);

using namespace std;
using namespace nba;

namespace nba {

static void coproc_task_input_cb(struct ev_loop *loop, struct ev_async *watcher, int revents)
{
    struct coproc_thread_context *ctx = (struct coproc_thread_context *) ev_userdata(loop);
    OffloadTask *task = nullptr;
    int ret;
    #ifdef USE_NVPROF
    nvtxRangePush("task_input_cb");
    #endif
    //uint64_t l = rte_ring_count(ctx->task_input_queue);
    //print_ratelimit("# coproc overlap chances", l, 100);
    /* To multiplex multiple streams, we process only one task and postpone
     * processing of the remainings.  The later steps have higher priority
     * and libev will call them first and then call earlier steps again. */
    ret = rte_ring_dequeue(ctx->task_input_queue, (void **) &task);
    if (task != nullptr) {
        task->coproc_ctx = ctx;
        task->copy_h2d();
        task->execute();
        /* We separate d2h copy step since CUDA implicitly synchronizes
         * kernel executions. See more details at:
         * http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#implicit-synchronization */
        ctx->d2h_pending_queue->push_back(task);
        ev_feed_event(loop, ctx->task_d2h_watcher, EV_ASYNC);
    }
    /* Let libev to call this handler again if we have remaining tasks.
     * ev_feed_event() is a very light-weight call as it does not do any
     * synchronization, unlike ev_async_send().  It just marks the event as
     * pending for the next iteration. */
    uint64_t input_queue_len = rte_ring_count(ctx->task_input_queue);
    if (input_queue_len > 0 && !ev_is_pending(watcher) && !ctx->loop_broken)
        ev_feed_event(loop, watcher, EV_ASYNC);
    #ifdef USE_NVPROF
    nvtxRangePop();
    #endif
}

static void coproc_task_d2h_cb(struct ev_loop *loop, struct ev_async *watcher, int revents)
{
    struct coproc_thread_context *ctx = (struct coproc_thread_context *) ev_userdata(loop);
    #ifdef USE_NVPROF
    nvtxRangePush("task_d2h_cb");
    #endif
    if (ctx->d2h_pending_queue->size() > 0) {
        OffloadTask *task = ctx->d2h_pending_queue->front();
        ctx->d2h_pending_queue->pop_front();
        if (task->poll_kernel_finished()) {
            //task->cctx->sync();
            task->copy_d2h();
            ctx->task_done_queue->push_back(task);
            if (ctx->task_done_queue->size() >= NBA_MAX_KERNEL_OVERLAP || !ev_is_pending(ctx->task_input_watcher))
                ev_feed_event(loop, ctx->task_done_watcher, EV_ASYNC);
        } else
            ctx->d2h_pending_queue->push_back(task);
    }
    if (ctx->d2h_pending_queue->size() > 0 && !ev_is_pending(watcher) && !ctx->loop_broken)
        ev_feed_event(loop, watcher, EV_ASYNC);
    #ifdef USE_NVPROF
    nvtxRangePop();
    #endif
}

static void coproc_task_done_cb(struct ev_loop *loop, struct ev_async *watcher, int revents)
{
    struct coproc_thread_context *ctx = (struct coproc_thread_context *) ev_userdata(loop);
    #ifdef USE_NVPROF
    nvtxRangePush("task_done_cb");
    #endif
    if (ctx->task_done_queue->size() > 0) {
        OffloadTask *task = ctx->task_done_queue->front();
        ctx->task_done_queue->pop_front();
        if (task->poll_d2h_copy_finished()) {
            task->notify_completion();
        } else
            ctx->task_done_queue->push_back(task);
    }
    if (ctx->task_done_queue->size() > 0 && !ev_is_pending(watcher) && !ctx->loop_broken)
        ev_feed_event(loop, watcher, EV_ASYNC);
    #ifdef USE_NVPROF
    nvtxRangePop();
    #endif
}

static void coproc_terminate_cb(struct ev_loop *loop, struct ev_async *watcher, int revents)
{
    struct coproc_thread_context *ctx = (struct coproc_thread_context *) ev_userdata(loop);
    /* Try to finish all pending tasks first. */
    ev_invoke_pending(loop);
    /* Break the loop. */
    ctx->loop_broken = true;
    ev_break(loop, EVBREAK_ALL);
}

static void coproc_init_offloadable_cb(struct ev_loop *loop, struct ev_async *watcher, int revents)
{
    return;
}

void *coproc_loop(void *arg)
{
    struct coproc_thread_context *ctx = (struct coproc_thread_context *) arg;
    assert(0 == rte_malloc_validate(ctx, NULL));

    /* Fool DPDK that we are on their lcore abstraction. */
    RTE_PER_LCORE(_lcore_id) = ctx->loc.core_id;
    RTE_PER_LCORE(_socket_id) = rte_lcore_to_socket_id(ctx->loc.core_id);

    /* Ensure we are on the right core. */
    assert(rte_socket_id() == ctx->loc.node_id);
    assert(rte_lcore_id() == ctx->loc.core_id);

    char temp[64];
    snprintf(temp, 64, "coproc.%u:%u@%u", ctx->loc.node_id, ctx->loc.local_thread_idx, ctx->loc.core_id);
    prctl(PR_SET_NAME, temp, 0, 0, 0);
    threading::bind_cpu(ctx->loc.core_id);
    #ifdef USE_NVPROF
    nvtxNameOsThread(pthread_self(), temp);
    #endif

    /* Initialize task queues. */
    ctx->d2h_pending_queue = new FixedRing<OffloadTask *, nullptr>(256, ctx->loc.node_id);
    ctx->task_done_queue   = new FixedRing<OffloadTask *, nullptr>(256, ctx->loc.node_id);

    /* Initialize the event loop. */
    ctx->loop = ev_loop_new(EVFLAG_AUTO);
    ctx->loop_broken = false;
    ev_set_userdata(ctx->loop, ctx);

    /* Register the termination event. */
    ev_set_cb(ctx->terminate_watcher, coproc_terminate_cb);
    ev_async_start(ctx->loop, ctx->terminate_watcher);

    size_t num_ctx_per_device = ctx->num_comp_threads_per_node * system_params["COPROC_CTX_PER_COMPTHREAD"];

    if (dummy_device) {
        new (ctx->device) DummyComputeDevice(ctx->loc.node_id, ctx->device_id, num_ctx_per_device);
    } else {
        #if defined(USE_CUDA) && defined(USE_PHI)
            #error "Simultaneous running of CUDA and Phi is not supported yet."
        #endif
        // TODO: replace here with factory pattern
        #ifdef USE_CUDA
        new (ctx->device) CUDAComputeDevice(ctx->loc.node_id, ctx->device_id, num_ctx_per_device);
        #endif
        #ifdef USE_PHI
        new (ctx->device) PhiComputeDevice(ctx->loc.node_id, ctx->device_id, num_ctx_per_device);
        #endif
    }

    /* Register the task input watcher. */
    ctx->task_done_watcher = new struct ev_async;
    ctx->task_d2h_watcher = new struct ev_async;
    ev_async_init(ctx->task_input_watcher, coproc_task_input_cb);
    ev_async_init(ctx->task_d2h_watcher, coproc_task_d2h_cb);
    ev_async_init(ctx->task_done_watcher, coproc_task_done_cb);
    ev_set_priority(ctx->task_input_watcher, EV_MAXPRI);
    ev_set_priority(ctx->task_d2h_watcher, EV_MAXPRI - 1);
    ev_set_priority(ctx->task_done_watcher, EV_MAXPRI - 2);
    ev_async_start(ctx->loop, ctx->task_input_watcher);
    ev_async_start(ctx->loop, ctx->task_d2h_watcher);
    ev_async_start(ctx->loop, ctx->task_done_watcher);

    RTE_LOG(NOTICE, COPROC, "@%u: initialized device %s\n", ctx->loc.core_id, ctx->device->type_name.c_str());
    ctx->thread_init_done_barrier->proceed();

    ctx->offloadable_init_barrier->wait();
    if (ctx->comp_ctx_to_init_offloadable != NULL) {
        RTE_LOG(INFO, COPROC, "@%u: initializing offloadable elements for %s\n",
                ctx->loc.core_id, ctx->device->type_name.c_str());
        comp_thread_context *comp_ctx = ctx->comp_ctx_to_init_offloadable;
        comp_ctx->initialize_offloadables_per_node(ctx->device);
    }
    ctx->offloadable_init_done_barrier->proceed();

    ctx->loopstart_barrier->wait();
    RTE_LOG(NOTICE, COPROC, "@%u: starting event loop\n", ctx->loc.core_id);

    /* The coprocessor event loop. */
    ev_run(ctx->loop, 0);

    // FIXME: delay the following operations after termination of IO threads
    //ctx->device->~ComputeDevice();
    //rte_free(ctx->device);
    rte_free(arg);
    return NULL;
}

}

// vim: ts=8 sts=4 sw=4 et foldmethod=marker
