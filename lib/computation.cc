/**
 * NBA's Computation Thread.
 *
 * Author: Joongi Kim <joongi@an.kaist.ac.kr>
 */

#include "config.hh"
#include "log.hh"
#include "types.hh"
#include "thread.hh"
#include "element.hh"
#include "elementgraph.hh"
#include "element_map.hh"
#include "loadbalancer.hh"
#include "computation.hh"
#include "annotation.hh"
#include "computecontext.hh"
extern "C" {
#include <unistd.h>
#include <numa.h>
#include <sys/prctl.h>
#include <rte_config.h>
#include <rte_common.h>
#include <rte_log.h>
#include <rte_errno.h>
#include <rte_malloc.h>
#include <rte_mbuf.h>
#include <rte_ring.h>
#include <rte_cycles.h>
#include <rte_prefetch.h>
#include <rte_per_lcore.h>
#include <ev.h>
#include <click_parser.h>
}
#include <exception>
#include <stdexcept>

using namespace std;
using namespace nba;

static thread_local uint64_t recv_batch_cnt = 0;
RTE_DECLARE_PER_LCORE(unsigned, _lcore_id);

namespace nba {

static void comp_terminate_cb(struct ev_loop *loop, struct ev_async *watcher, int revents)
{
    ev_break(loop, EVBREAK_ALL);
}

static void comp_offload_task_completion_cb(struct ev_loop *loop, struct ev_async *watcher, int revents)
{
    comp_thread_context *ctx = (comp_thread_context *) ev_userdata(loop);
    OffloadTask *tasks[ctx->task_completion_queue_size];
    unsigned nr_tasks = rte_ring_sc_dequeue_burst(ctx->task_completion_queue,
                                                  (void **) tasks,
                                                  ctx->task_completion_queue_size);
    print_ratelimit("avg.# tasks completed", nr_tasks, 10000);
    for (unsigned t = 0; t < nr_tasks; t++) {
        /* We already finished postprocessing.
         * Retrieve the task and results. */
        OffloadTask *task = tasks[t];

        /* Return the context. */
        task->cctx->state = ComputeContext::READY;
        ctx->cctx_list.push_back(task->cctx);

        Element *last_elem = task->offloaded_elements[0]; // TODO: get last one if multiple elements

        /* Run next element graph. */
        for (size_t b = 0; b < task->num_batches; b ++) {
            ctx->elem_graph->enqueue_postproc_batch(task->batches[b], last_elem,
                                                    task->input_ports[b]);
        }

        /* Free the task object. */
        // TODO: generalize device index
        ctx->inspector->dev_finished_batch_count[0] += task->num_batches;
        rte_mempool_put(ctx->task_pool, (void *) task);
    }

    /* Let the computation thread to receive packets again. */
    ctx->resume_rx();
    if (ev_async_pending(ctx->rx_watcher))
        ev_invoke(ctx->loop, ctx->rx_watcher, EV_ASYNC);
    else {}
        //kill(ctx->io_ctx->thread_id, SIGUSR1);
}

static void comp_packetbatch_init(struct rte_mempool *mp, void *arg, void *obj, unsigned idx)
{
    PacketBatch *b = (PacketBatch*) obj;
    new (b) PacketBatch();
}

static void comp_task_init(struct rte_mempool *mp, void *arg, void *obj, unsigned idx)
{
    OffloadTask *t = (OffloadTask *) obj;
    new (t) OffloadTask();
}

comp_thread_context::comp_thread_context() {
    terminate_watcher = nullptr;
    thread_init_barrier = nullptr;
    ready_cond = nullptr;
    ready_flag = nullptr;
    elemgraph_lock = nullptr;
    node_local_storage = nullptr;

    num_tx_ports = 0;
    num_nodes = 0;
    num_combatch_size = 0;
    num_batchpool_size = 0;
    num_taskpool_size = 0;
    num_comp_ppdepth = 0;
    num_coproc_ppdepth = 0;
    rx_queue_size = 0;
    rx_wakeup_threshold = 0;

    batch_pool = nullptr;
    task_pool = nullptr;
    elem_graph = nullptr;
    input_batch = nullptr;

    io_ctx = nullptr;
    named_offload_devices = nullptr;
    offload_devices = nullptr;
    for (unsigned i = 0; i < NBA_MAX_COPROCESSORS; i++) {
        offload_input_queues[i] = nullptr;
    }

    task_completion_queue   = nullptr;
    task_completion_watcher = nullptr;
}


comp_thread_context::~comp_thread_context() {
    delete elem_graph;
}

void comp_thread_context::stop_rx()
{
    ev_async_stop(loop, rx_watcher);
}

void comp_thread_context::resume_rx()
{
    /* Flush the processing pipeline. */
    elem_graph->flush_delayed_batches();

    /* Reactivate the packet RX event. */
    ev_async_start(loop, rx_watcher);
}

static void *click_module_handler(int global_idx, const char* name, int argc, char **argv, void *priv)
{
    comp_thread_context *ctx = (comp_thread_context *) priv;
    string elem_name(name);
    if (element_registry.find(elem_name) == element_registry.end()) {
        rte_panic("click_module_handler(): element with name \"%s\" does not exist.\n", name);
    }
    Element *el = element_registry[elem_name].instantiate();

    vector<string> args;
    for (int i = 0; i < argc; i++)
        args.push_back(string(argv[i]));
    el->configure(ctx, args);

    ctx->elem_graph->add_element(el);
    return el;
}

static void click_module_linker(void *from, int from_output, void *to, int to_input, void *priv)
{
    comp_thread_context *ctx = (comp_thread_context *) priv;
    Element *from_module = (Element *) from;
    Element *to_module   = (Element *) to;
    ctx->elem_graph->link_element(to_module, to_input, from_module, from_output);

    if ((from_module->get_type() & ELEMTYPE_OFFLOADABLE) != 0 &&
        (to_module->get_type() & ELEMTYPE_OFFLOADABLE) != 0) {
        RTE_LOG(INFO, COMP, "Found consequent offloadable modules %s and %s\n",
                from_module->class_name(), to_module->class_name());
    }
}

void comp_thread_context::build_element_graph(const char* config_file) {

    elemgraph_lock->acquire();

    FILE* input = fopen(config_file, "r");
    ParseInfo *pi = click_parse_configuration(input, click_module_handler, click_module_linker, this);

    /* Schedulable elements will be automatically detected during addition.
     * (They corresponds to multiple "root" elements.) */

    click_destroy_configuration(pi);
    fclose(input);

    elemgraph_lock->release();
}

void comp_thread_context::initialize_graph_global() {
    elemgraph_lock->acquire();
    /* globally once invocation is guaranteed by main.cc */
    for (Element *el : elem_graph->get_elements()) {
        el->initialize_global();
    }
    elemgraph_lock->release();
}

void comp_thread_context::initialize_graph_per_node() {
    elemgraph_lock->acquire();
    /* per-node invocation is guaranteed by main.cc */
    for (Element *el : elem_graph->get_elements()) {
        el->initialize_per_node();
    }
    elemgraph_lock->release();
}

void comp_thread_context::initialize_offloadables_per_node(ComputeDevice *device) {
    elemgraph_lock->acquire();
    for (Element *el : elem_graph->get_elements()) {
        OffloadableElement *elem = dynamic_cast<OffloadableElement *> (el);
        if (elem == nullptr)
            continue;
        string dev_name = device->type_name;
        RTE_LOG(DEBUG, COMP, "initializing offloadable element %s for device %s at node %u\n",
                elem->class_name(), dev_name.c_str(), this->loc.node_id);
        if (elem->offload_init_handlers.find(dev_name) == elem->offload_init_handlers.end())
            continue;
        elem->offload_init_handlers[dev_name](device);
    }
    elemgraph_lock->release();
}


void comp_thread_context::initialize_graph_per_thread() {
    // per-element configuration
    for (Element *el : elem_graph->get_elements()) {
        el->initialize();
    }
}

void comp_thread_context::io_tx_new(void* data, size_t len, int out_port)
{
    if (len > NBA_MAX_PACKET_SIZE) {
        RTE_LOG(WARNING, COMP, "io_tx_new(): Too large packet!\n");
        return;
    }

    struct new_packet* new_packet = 0;
    int ret = rte_mempool_get(this->io_ctx->new_packet_request_pool, (void**)&new_packet);
    if (ret != 0) {
        RTE_LOG(WARNING, COMP, "io_tx_new(): No more buffer!\n");
        return;
    }

    new_packet->len = len;
    new_packet->out_port = out_port;
    memcpy(new_packet->buf, data, len);

    ret = rte_ring_enqueue(this->io_ctx->new_packet_request_ring, new_packet);
    assert(ret == 0);
}

}

// vim: ts=8 sts=4 sw=4 et foldmethod=marker
