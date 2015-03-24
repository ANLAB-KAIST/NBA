/**
 * nShader's Computation Thread.
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
#include "graphanalysis.hh"

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
extern "C"{
#include <ev.h>
#include <click_parser.h>
}

#include <exception>
#include <stdexcept>

using namespace std;
using namespace nshader;

static thread_local uint64_t recv_batch_cnt = 0;
RTE_DECLARE_PER_LCORE(unsigned, _lcore_id);

namespace nshader {

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
    for (unsigned i = 0; i < NSHADER_MAX_COPROCESSORS; i++) {
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
    Element *module = element_registry[elem_name].instantiate();

    vector<string> args;
    for (int i = 0; i < argc; i++)
        args.push_back(string(argv[i]));
    module->configure(ctx, args);

    ctx->elem_graph->add_element(module);
#if 0
    std::vector<int> my_datablocks;
    module->get_datablocks(my_datablocks);

    for(auto block_id : my_datablocks)
    {
    	struct read_roi_info read_roi;
    	struct write_roi_info write_roi;
    	memset(&read_roi, 0, sizeof(read_roi));
    	memset(&write_roi, 0, sizeof(write_roi));

    	ctx->datablock_registry[block_id]->get_read_roi(&read_roi);
    	ctx->datablock_registry[block_id]->get_write_roi(&write_roi);

    	L::Bitmap read_bitmap(2048);
    	L::Bitmap write_bitmap(2048);

    	if(read_roi.type == READ_PARTIAL_PACKET)
    		read_bitmap.setRange(true, read_roi.offset, read_roi.offset+read_roi.length);
    	else if(read_roi.type == READ_NONE)
    	{

    	}
    	else
    	{
    		read_bitmap.setRange(true, 0, 2048);
    	}

    	if(write_roi.type == WRITE_PARTIAL_PACKET)
    		write_bitmap.setRange(true, write_roi.offset, write_roi.offset+write_roi.length);
    	else if(write_roi.type == WRITE_NONE || write_roi.type == WRITE_FIXED_SEGMENTS)
    	{

    	}
    	else
    	{
    		write_bitmap.setRange(true, 0, 2048);
    	}

    	//printf("%s's read roi: ", ctx->datablock_registry[block_id]->name());
    	//read_bitmap.print();
    	//printf("%s's write roi: ", ctx->datablock_registry[block_id]->name());
    	//write_bitmap.print();

    	module->addROI(block_id, read_bitmap, write_bitmap);
    }
#endif
    return module;
}

static void click_module_linker(void *from, int from_output, void *to, int to_input, void *priv)
{
    comp_thread_context *ctx = (comp_thread_context *) priv;
    Element *from_module = (Element *) from;
    Element *to_module   = (Element *) to;
    ctx->elem_graph->link_element(to_module, to_input, from_module, from_output);

    from_module->link(to_module);

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
    int num_modules = click_num_module(pi);
    /* Schedulable elements will be automatically detected during addition.
     * (They corresponds to multiple "root" elements.) */
    GraphAnalysis::analyze(pi);
    for(int k=0; k<num_modules; k++)
    {
    	Element* elem = (Element*)click_get_module(pi, k);

    	if(elem->getLinearGroup() >= 0)
    	{
    		RTE_LOG(INFO, COMP, "Element [%s] is in group %d\n",
    				elem->class_name(), elem->getLinearGroup());
    	}
    }

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
    if (len > NSHADER_MAX_PACKET_SIZE) {
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
