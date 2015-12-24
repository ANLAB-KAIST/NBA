/**
 * NBA's Computation Thread.
 *
 * Author: Joongi Kim <joongi@an.kaist.ac.kr>
 */

#include <nba/core/intrinsic.hh>
#include <nba/core/threading.hh>
#include <nba/framework/config.hh>
#include <nba/framework/logging.hh>
#include <nba/framework/threadcontext.hh>
#include <nba/framework/loadbalancer.hh>
#include <nba/framework/computation.hh>
#include <nba/framework/computecontext.hh>
#include <nba/framework/graphanalysis.hh>
#include <nba/framework/elementgraph.hh>
#include <nba/element/element.hh>
#include <nba/element/element_map.hh>
#include <nba/element/annotation.hh>

#include <unistd.h>
#include <numa.h>
#include <sys/prctl.h>
#include <rte_config.h>
#include <rte_common.h>
#include <rte_errno.h>
#include <rte_malloc.h>
#include <rte_mbuf.h>
#include <rte_mempool.h>
#include <rte_ring.h>
#include <rte_cycles.h>
#include <rte_prefetch.h>
#include <rte_per_lcore.h>
extern "C" {
#include <ev.h>
#include <click_parser.h>
}

#include <algorithm>
#include <exception>
#include <stdexcept>

using namespace std;
using namespace nba;

static thread_local uint64_t recv_batch_cnt = 0;
RTE_DECLARE_PER_LCORE(unsigned, _lcore_id);

namespace nba
{

comp_thread_context::comp_thread_context()
{
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
    num_coproc_ppdepth = 0;

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


comp_thread_context::~comp_thread_context()
{
    delete elem_graph;
}

void comp_thread_context::stop_rx()
{
    ev_async_stop(loop, rx_watcher);
}

void comp_thread_context::resume_rx()
{
    /* Flush the processing pipeline. */
    elem_graph->flush_tasks();

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

    for (auto block_id : my_datablocks) {
        struct read_roi_info read_roi;
        struct write_roi_info write_roi;
        memset(&read_roi, 0, sizeof(read_roi));
        memset(&write_roi, 0, sizeof(write_roi));
        ctx->datablock_registry[block_id]->get_read_roi(&read_roi);
        ctx->datablock_registry[block_id]->get_write_roi(&write_roi);

        L::Bitmap read_bitmap(2048);
        L::Bitmap write_bitmap(2048);
        if (read_roi.type == READ_PARTIAL_PACKET) {
            read_bitmap.setRange(true, read_roi.offset, read_roi.offset+read_roi.length);
        } else if (read_roi.type == READ_NONE) {

        } else {
            read_bitmap.setRange(true, 0, 2048);
        }

        if (write_roi.type == WRITE_PARTIAL_PACKET) {
            write_bitmap.setRange(true, write_roi.offset, write_roi.offset+write_roi.length);
        } else if (write_roi.type == WRITE_NONE || write_roi.type == WRITE_FIXED_SEGMENTS) {

        } else {
            write_bitmap.setRange(true, 0, 2048);
        }
        module->add_roi(block_id, read_bitmap, write_bitmap);
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
}

void comp_thread_context::build_element_graph(const char* config_file)
{
    elemgraph_lock->acquire();

    FILE* input = fopen(config_file, "r");

    /* Parse the config file and build the element graph object. */
    ParseInfo *pi = click_parse_configuration(input, click_module_handler, click_module_linker, this);
    int num_modules = click_num_module(pi);

    /* Schedulable elements will be automatically detected during addition.
     * (They corresponds to multiple "root" elements.) */
    GraphAnalyzer ga;
    ga.analyze(pi);
    auto& linear_groups = ga.get_linear_groups();

    /* Print the assigned group of each element. */
    for(int k = 0; k < num_modules; k++) {
        Element* elem = (Element*) click_get_module(pi, k);
        if (elem->get_linear_group() >= 0) {
            RTE_LOG(INFO, ELEM, "Element [%s] is in group %d\n",
                    elem->class_name(), elem->get_linear_group());
        }
    }
    RTE_LOG(INFO, ELEM, "Number of linear groups: %lu\n", linear_groups.size());
    for (vector<GraphMetaData *> group : linear_groups) {

        Element *prev_el = nullptr;
        OffloadableElement *prev_oel = nullptr;
        set<int> all_new_dbids, all_deleted_dbids;

        for (GraphMetaData *m : group) {

            int dbids[NBA_MAX_DATABLOCKS];
            vector<int> cur_dbids(NBA_MAX_DATABLOCKS, -1), prev_dbids(NBA_MAX_DATABLOCKS, -1);
            vector<int> new_dbids(NBA_MAX_DATABLOCKS, -1), deleted_dbids(NBA_MAX_DATABLOCKS, -1);
            size_t num_db;
            Element *el = (Element *) m;
            OffloadableElement *oel = dynamic_cast<OffloadableElement *>(m);
            if (oel != nullptr) {
                num_db = oel->get_used_datablocks(dbids);
                for (size_t d = 0; d < num_db; d++)
                    cur_dbids.push_back(dbids[d]);
                sort(cur_dbids.begin(), cur_dbids.end());
            }
            if (prev_oel != nullptr) {
                num_db = prev_oel->get_used_datablocks(dbids);
                for (size_t d = 0; d < num_db; d++)
                    prev_dbids.push_back(dbids[d]);
                sort(prev_dbids.begin(), prev_dbids.end());
            }

            new_dbids.reserve(cur_dbids.size() + prev_dbids.size());
            auto it = set_difference(cur_dbids.begin(), cur_dbids.end(),
                                     prev_dbids.begin(), prev_dbids.end(),
                                     new_dbids.begin());
            new_dbids.resize(it - new_dbids.begin());
            deleted_dbids.reserve(cur_dbids.size() + prev_dbids.size());
            it      = set_difference(prev_dbids.begin(), prev_dbids.end(),
                                     cur_dbids.begin(), cur_dbids.end(),
                                     deleted_dbids.begin());
            deleted_dbids.resize(it - deleted_dbids.begin());

            auto& actions = this->elem_graph->offl_actions;
            auto& fin     = this->elem_graph->offl_fin;
            for (int dbid : new_dbids) {
                auto key = make_pair(oel, dbid);
                auto ret = actions.find(key);
                if (ret == actions.end()) {
                    actions.insert(make_pair(key, ELEM_OFFL_PREPROC));
                } else {
                    int flag = actions[key];
                    actions[key] = flag | ELEM_OFFL_PREPROC;
                }
                RTE_LOG(INFO, ELEM, "%s (%p, %d) -> preproc\n", el->class_name(), oel, dbid);
                all_new_dbids.insert(dbid);
            }
            for (int dbid : deleted_dbids) {
                auto key = make_pair(prev_oel, dbid);
                auto ret = actions.find(key);
                if (ret == actions.end()) {
                    actions.insert(make_pair(key, ELEM_OFFL_POSTPROC));
                } else {
                    int flag = actions[key];
                    actions[key] = flag | ELEM_OFFL_POSTPROC;
                }
                RTE_LOG(INFO, ELEM, "%s (%p, %d) -> postproc\n", prev_el->class_name(), prev_oel, dbid);
                all_deleted_dbids.insert(dbid);

                if (all_new_dbids.size() > 0
                        && includes(all_new_dbids.begin(), all_new_dbids.end(),
                                    all_deleted_dbids.begin(), all_deleted_dbids.end())
                        && includes(all_deleted_dbids.begin(), all_deleted_dbids.end(),
                                    all_new_dbids.begin(), all_new_dbids.end())) {
                    /* When all used datablocks are postprocessed... */
                    fin.insert(prev_oel);
                    RTE_LOG(INFO, ELEM, "%s (%p) -> clear\n", prev_el->class_name(), prev_oel);
                    all_new_dbids.clear();
                    all_deleted_dbids.clear();
                }
            }

            prev_el = el;
            prev_oel = oel;
        }
        // TODO: 여기서 한번 더 체크해줘야 offloadable이 맨 마지막인 경우 제대로 처리될 것.
    }
    click_destroy_configuration(pi);
    fclose(input);
    elemgraph_lock->release();
}

void comp_thread_context::initialize_graph_global()
{
    elemgraph_lock->acquire();
    /* globally once invocation is guaranteed by main.cc */
    for (Element *el : elem_graph->get_elements()) {
        el->initialize_global();
    }
    elemgraph_lock->release();
}

void comp_thread_context::initialize_graph_per_node()
{
    elemgraph_lock->acquire();
    /* per-node invocation is guaranteed by main.cc */
    for (Element *el : elem_graph->get_elements()) {
        el->initialize_per_node();
    }
    elemgraph_lock->release();
}

void comp_thread_context::initialize_offloadables_per_node(ComputeDevice *device)
{
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


void comp_thread_context::initialize_graph_per_thread()
{
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
