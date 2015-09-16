#include <nba/core/offloadtypes.hh>
#include <nba/core/vector.hh>
#include <nba/framework/config.hh>
#include <nba/framework/elementgraph.hh>
#include <nba/framework/offloadtask.hh>
#include <nba/element/packet.hh>
#include <nba/element/packetbatch.hh>
#include <nba/element/element.hh>
#include <string>
#include <vector>
#include <cassert>
#include <cstdint>
#include <climits>
#include <rte_config.h>
#include <rte_common.h>
#include <rte_memory.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_prefetch.h>
#include <rte_branch_prediction.h>
#include <rte_cycles.h>

using namespace std;
using namespace nba;
 
static uint64_t task_id = 0;

Element::Element() : next_elems(), next_connected_inputs()
{
    num_min_inputs = num_max_inputs = 0;
    num_min_outputs = num_max_outputs = 0;
    memset(branch_count, 0, sizeof(uint64_t) * ElementGraph::num_max_outputs);
    for (int i = 0; i < ElementGraph::num_max_outputs; i++)
        outputs[i] = OutputPort(this, i);
}

Element::~Element()
{
}

int Element::_process_batch(int input_port, PacketBatch *batch)
{
    memset(output_counts, 0, sizeof(uint16_t) * ElementGraph::num_max_outputs);
    batch->has_dropped = false;
    batch->drop_count = 0;
//#define NBA_LOOP_UNROLLING
#ifdef NBA_LOOP_UNROLLING
    #define NBA_UNROLL_STRIDE (4)
    unsigned stride;
    for (stride = 0; stride < batch->count; stride += NBA_UNROLL_STRIDE) {
        unsigned idx1 = stride + 0;
        unsigned idx2 = stride + 1;
        unsigned idx3 = stride + 2;
        unsigned idx4 = stride + 3;
        Packet *pkt1 = Packet::from_base(batch->packets[idx1]);
        Packet *pkt2 = Packet::from_base(batch->packets[idx2]);
        Packet *pkt3 = Packet::from_base(batch->packets[idx3]);
        Packet *pkt4 = Packet::from_base(batch->packets[idx4]);
        pkt1->bidx = idx1;
        pkt2->bidx = idx2;
        pkt3->bidx = idx3;
        pkt4->bidx = idx4;
        this->process(input_port, pkt1);
        this->process(input_port, pkt2);
        this->process(input_port, pkt3);
        this->process(input_port, pkt4);
    }
    for (unsigned p = stride - NBA_UNROLL_STRIDE; p < batch->count; p++) {
        Packet *pkt = Packet::from_base(batch->packets[p]);
        pkt->bidx = p;
        this->process(input_port, pkt);
    }
    #undef NBA_UNROLL_STRIDE
#else
    for (unsigned p = 0; p < batch->count; p++) {
        if (likely(!batch->excluded[p])) {
            Packet *pkt = Packet::from_base(batch->packets[p]);
            pkt->bidx = p;
            this->process(input_port, pkt);
        }
    }
#endif
    if (batch->has_dropped)
        batch->collect_excluded_packets();
    batch->has_results = true;
    return 0; // this value will be ignored.
}

int VectorElement::_process_batch(int input_port, PacketBatch *batch)
{
    batch->has_dropped = false;
    batch->drop_count = 0;
    unsigned stride = 0;
    for (stride = 0; stride < batch->count; stride += NBA_VECTOR_WIDTH) {
        vec_mask_t mask = _mm256_set1_epi64x(1);
        vec_mask_arg_t mask_arg;
        _mm256_storeu_si256((__m256i *) &mask_arg, mask);
        Packet *pkt_vec[NBA_VECTOR_WIDTH] = {
            Packet::from_base(batch->packets[stride + 0]),
            Packet::from_base(batch->packets[stride + 1]),
            Packet::from_base(batch->packets[stride + 2]),
            Packet::from_base(batch->packets[stride + 3]),
            Packet::from_base(batch->packets[stride + 4]),
            Packet::from_base(batch->packets[stride + 5]),
            Packet::from_base(batch->packets[stride + 6]),
            Packet::from_base(batch->packets[stride + 7]),
        };
        // TODO: vectorize?
        pkt_vec[0]->bidx = stride + 0;
        pkt_vec[1]->bidx = stride + 1;
        pkt_vec[2]->bidx = stride + 2;
        pkt_vec[3]->bidx = stride + 3;
        pkt_vec[4]->bidx = stride + 4;
        pkt_vec[5]->bidx = stride + 5;
        pkt_vec[6]->bidx = stride + 6;
        pkt_vec[7]->bidx = stride + 7;
        this->process_vector(input_port, pkt_vec, mask_arg);
    }
    {
        vec_mask_arg_t mask_arg = {0,};
        Packet *pkt_vec[NBA_VECTOR_WIDTH] = {nullptr,};
        for (unsigned i = stride - NBA_VECTOR_WIDTH, j = 0; i < batch->count; i++, j++) {
            pkt_vec[j] = Packet::from_base(batch->packets[i]);
            pkt_vec[j]->bidx = i;
            mask_arg.m[j] = 1;
        }
        this->process_vector(input_port, pkt_vec, mask_arg);
    }
    if (batch->has_dropped)
        batch->collect_excluded_packets();
    batch->has_results = true;
    return 0;
}

int PerBatchElement::_process_batch(int input_port, PacketBatch *batch)
{
    int ret = this->process_batch(input_port, batch);
    batch->has_results = true;
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
    if (num_max_outputs > NBA_MAX_ELEM_NEXTS) {
        rte_panic("Element::update_port_count(): Too many possible output ports (max: %d)\n", NBA_MAX_ELEM_NEXTS);
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

int OffloadableElement::offload(ElementGraph *mother, PacketBatch *in_batch, int input_port)
{
    int dev_idx = 0;
    OffloadTask *task = nullptr;
    if (tasks[dev_idx] == nullptr) {
        #ifdef USE_NVPROF
        nvtxRangePush("accum_batch");
        #endif
        /* We assume: task pool size >= task input queue length */
        int ret = rte_mempool_get(ctx->task_pool, (void **) &task);
        if (ret == -ENOENT) {
            //if (!ctx->io_ctx->loop_broken)
            //    ev_run(ctx->io_ctx->loop, EVRUN_NOWAIT);
            /* Keep the current batch for later processing. */
            return -1;
        }
        new (task) OffloadTask();
        task->state = TASK_INITIALIZING;
        task->task_id = task_id ++;
        task->src_loop = ctx->loop;
        task->comp_ctx = ctx;
        task->completion_queue = ctx->task_completion_queue;
        task->completion_watcher = ctx->task_completion_watcher;
        task->elemgraph = mother;
        task->local_dev_idx = dev_idx;
        //task->device = ctx->offload_devices->at(dev_idx);
        //assert(task->device != nullptr);
        task->elem = this;

        tasks[dev_idx] = task;
    } else
        task = tasks[dev_idx];
    assert(task != nullptr);

    /* Add the current batch if the task batch is not full. */
    if (task->batches.size() < ctx->num_coproc_ppdepth) {
        task->batches.push_back(in_batch);
        task->input_ports.push_back(input_port);
        #ifdef USE_NVPROF
        nvtxMarkA("add_batch");
        #endif
    } else {
        return -1;
    }

    /* Start offloading when one or more following conditions are met:
     *  1. The task batch size has reached the limit (num_coproc_ppdepth).
     *     This is the ideal case that pipelining happens and GPU utilization is high.
     *  2. The load balancer has changed its decision to CPU.
     *     We need to flush the pending offload tasks.
     *  3. The time elapsed since the first packet is greater than
     *     10 x avg.task completion time.
     */
    assert(task->batches.size() > 0);
    uint64_t now = rte_rdtsc();
    if (task->batches.size() == ctx->num_coproc_ppdepth
        // || (task->num_bytes >= 64 * ctx->num_coproc_ppdepth * ctx->io_ctx->num_iobatch_size)
        // || (task->batches.size() > 1 && (rdtsc() - task->offload_start) / (double) rte_get_tsc_hz() > 0.0005)
        // || (ctx->io_ctx->mode == IO_EMUL && !ctx->stop_task_batching)
        )
    {
        //printf("avg task completion time: %.6f sec\n", ctx->inspector->avg_task_completion_sec[dev_idx]);

        tasks[dev_idx] = nullptr;  // Let the element be able to take next pkts/batches.
        task->offload_start = now;

        task->state = TASK_INITIALIZED;
        mother->ready_tasks[dev_idx].push_back(task);
        #ifdef USE_NVPROF
        nvtxRangePop();
        #endif
    }
    return 0;
}

int OffloadableElement::enqueue_batch(PacketBatch *batch)
{
    finished_batches.push_back(batch);
    return 0;
}

int OffloadableElement::dispatch(uint64_t loop_count, PacketBatch*& out_batch, uint64_t &next_delay)
{
    /* Retrieve out_batch from the internal completion_queue. */
    if (finished_batches.size() > 0) {
        out_batch = finished_batches.front();
        finished_batches.pop_front();
    } else {
        out_batch = nullptr;
    }
    next_delay = 0;
    return 0;
}

void OffloadableElement::dummy_compute_handler(ComputeContext *ctx,
                                               struct resource_param *res)
{
    return;
}
// vim: ts=8 sts=4 sw=4 et
