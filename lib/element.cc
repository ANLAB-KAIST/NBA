#include "element.hh"
#include "elementgraph.hh"
#include "config.hh"
#include "packet.hh"
#include "packetbatch.hh"
extern "C" {
#include <rte_config.h>
#include <rte_common.h>
#include <rte_memory.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_prefetch.h>
#include <rte_branch_prediction.h>
#include <rte_cycles.h>
}
#include <string>
#include <vector>
#include <cassert>
#include <cstdint>
#include <climits>

using namespace std;
using namespace nba;

Element::Element() : next_elems(), next_connected_inputs()
{
    num_min_inputs = num_max_inputs = 0;
    num_min_outputs = num_max_outputs = 0;
    memset(branch_count,0,sizeof(uint64_t)*ElementGraph::num_max_outputs);
}

Element::~Element()
{
}

int Element::_process_batch(int input_port, PacketBatch *batch)
{
	//double batch_start = rte_rdtsc();
    //if (packet == nullptr)
    //    assert(0 == rte_mempool_get(ctx->packet_pool, (void **) &packet));
    //packet->set_mbuf_pool(ctx->packet_pool);
    for (unsigned p = 0; p < batch->count; p++) {
        if (likely(!batch->excluded[p])) {
            //packet->set_mbuf(packet, batch);
            batch->results[p] = this->process(input_port, batch->packets[p], &batch->annos[p]);
            //batch->results[p] = packet->get_results();
        }

    }
    batch->has_results = true;
    //double batch_end = rte_rdtsc();
    //batch->compute_time += (batch_end-batch_start) / batch->count;//ctx->num_combatch_size;
    return 0; // this value will be ignored.
}

int PerBatchElement::_process_batch(int input_port, PacketBatch *batch)
{
	//double batch_start = rte_rdtsc();
	int ret = this->process_batch(input_port, batch);
	batch->has_results = true;
	//double batch_end = rte_rdtsc();
	//batch->compute_time += (batch_end-batch_start) / batch->count;//ctx->num_combatch_size;
	return ret;
}

void Element::update_port_count()
{
    size_t i;

    string *port_str = new string(port_count());
    size_t delim_idx = port_str->find("/");
    assert(delim_idx != string::npos);

    string input_spec = port_str->substr(0, delim_idx);
    string output_spec = port_str->substr(delim_idx + 1, port_str->length() - delim_idx);

    size_t range_delim_idx = input_spec.find("-");
    if (range_delim_idx == string::npos) {
        num_min_inputs = num_max_inputs = atoi(input_spec.c_str());
        assert(num_min_inputs >= 0);
    } else {
        string range_left = input_spec.substr(0, range_delim_idx);
        string range_right = input_spec.substr(range_delim_idx, input_spec.length() - range_delim_idx);
        num_min_inputs = atoi(range_left.c_str());
        num_max_inputs = atoi(range_right.c_str());
    }

    range_delim_idx = output_spec.find("-");
    if (range_delim_idx == string::npos) {
        if (output_spec == "*") {
            num_min_outputs = 0;
            num_max_outputs = -1;
        } else {
            num_min_outputs = num_max_outputs = atoi(output_spec.c_str());
        }
    } else {
        string range_left = output_spec.substr(0, range_delim_idx);
        string range_right = output_spec.substr(range_delim_idx, output_spec.length() - range_delim_idx);
        num_min_outputs = atoi(range_left.c_str());
        num_max_outputs = atoi(range_right.c_str());
    }
    if (num_max_outputs != -1) {
        next_elems.reserve(num_max_outputs);
        next_connected_inputs.reserve(num_max_outputs);
    }

    delete port_str;
}

int Element::initialize_global() {
    return 0;
}

int Element::initialize_per_node() {
    return 0;
}

int Element::initialize() {
    return 0;
}

int Element::configure(comp_thread_context *ctx, vector<string> &args) {
    this->ctx = ctx;
    return 0;
}

void OffloadableElement::dummy_compute_handler(
        ComputeContext *ctx,
        struct resource_param *res,
        struct annotation_set **anno_ptr_array
) {
    return;
}
// vim: ts=8 sts=4 sw=4 et
