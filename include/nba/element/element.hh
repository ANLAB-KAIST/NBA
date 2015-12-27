#ifndef __NBA_ELEMENT_HH__
#define __NBA_ELEMENT_HH__

#include <nba/core/intrinsic.hh>
#include <nba/core/queue.hh>
#include <nba/core/offloadtypes.hh>
#include <nba/core/vector.hh>
#include <nba/framework/config.hh>
#include <nba/framework/graphanalysis.hh>
#include <nba/element/packet.hh>
#include <nba/element/packetbatch.hh>
#include <nba/element/nodelocalstorage.hh>
#include <string>
#include <vector>
#include <functional>
#include <unordered_map>
#include <cassert>

namespace nba {

/* Forward declarations. */
class Element;
class ElementGraph;
class OffloadTask;
class comp_thread_context;
class ComputeContext;
class ComputeDevice;

typedef std::function<void(ComputeContext *ctx, struct resource_param *res)> offload_compute_handler;
typedef std::function<void(ComputeDevice *dev)> offload_init_handler;

enum ElementType {
    /* PER_PACKET and PER_BATCH are exclusive to each other. */
    ELEMTYPE_PER_PACKET = 1,
    ELEMTYPE_PER_BATCH = 2,
    ELEMTYPE_SCHEDULABLE = 4,
    ELEMTYPE_OFFLOADABLE = 8,
    ELEMTYPE_INPUT = 16,
    ELEMTYPE_OUTPUT = 32,
    ELEMTYPE_VECTOR = 64,
};

struct element_info {
    int idx;
    /* NOTE: Non-mutable lambda expression of the same type is
     * converted to the function pointer by the compiler. */
    Element* (*instantiate)(void);
};

#define EXPORT_ELEMENT(...)

#define HANDLE_ALL_PORTS case 0: \
                         case 1: \
                         case 2: \
                         case 3

class Element : public GraphMetaData {
private:
    friend class Packet;

    class OutputPort {
        /** A simple utility class to emulate Click's output port. */

    private:
        friend class Element;
        friend class GraphMetaData;

        int my_idx;
        Element *elem;

        OutputPort() : my_idx(0), elem(nullptr) { }
        OutputPort(Element* elem, int idx) : my_idx(idx), elem(elem) { }
        virtual ~OutputPort() { }

    public:
        void push(Packet *pkt) const {
            /* We allow a packet to be pushed only once inside the process
             * handler.  If you want to push the same packet multiple times
             * to different outputs, you MUST clone it. */
            pkt->mother->results[pkt->bidx] = my_idx;
            if (pkt->cloned) {
                /* Store the cloned packet separately. */
                elem->output_cloned_packets[my_idx][elem->output_counts[my_idx]] = pkt;
            }
            elem->output_counts[my_idx] ++;
        }
    };

public:
    Element();
    virtual ~Element();

    inline const OutputPort &output(int idx) const { return outputs[idx]; }

    /* == User-defined properties and methods == */
    virtual const char *class_name() const = 0;
    virtual const char *port_count() const = 0;
    virtual int get_type() const { return ELEMTYPE_PER_PACKET; }

    virtual int configure(comp_thread_context *ctx, std::vector<std::string> &args);
    virtual int initialize() = 0;       // per-thread configuration. Called after coprocessor threads are initialized.
    virtual int initialize_global();    // thread-global configuration. Called before coprocessor threads are initialized.
    virtual int initialize_per_node();  // per-node configuration. Called before coprocessor threads are initialized.

    /** User-define function to process a packet. */
    virtual int process(int input_port, Packet *pkt) = 0;

    /**
     * A framework-internal base method always called by the element graph.
     */
    virtual int _process_batch(int input_port, PacketBatch *batch);

    virtual void get_datablocks(std::vector<int> &datablock_ids){}; //TODO fill here...

    comp_thread_context *ctx;

protected:
    /* Subclasses use below to manage node-local storage. */
    int num_nodes;
    int node_idx;

private:
    friend class ElementGraph;
    friend class Element::OutputPort;

    uint64_t branch_total = 0;
    uint64_t branch_miss = 0;
    uint64_t branch_count[NBA_MAX_ELEM_NEXTS];

    FixedArray<Element*, nullptr, NBA_MAX_ELEM_NEXTS> next_elems;
    FixedArray<int, -1, NBA_MAX_ELEM_NEXTS> next_connected_inputs;
    OutputPort outputs[NBA_MAX_ELEM_NEXTS];

    uint16_t output_counts[NBA_MAX_ELEM_NEXTS];
    Packet *output_cloned_packets[NBA_MAX_ELEM_NEXTS][NBA_MAX_COMP_BATCH_SIZE];

    /**
     * Parse "x/y" or "x1-x2/y1-y2" variations, where x is positive
     * integers and y is zero or positive integers or '*' (arbitrary
     * number).  This syntax follows Click's style.
     */
    void update_port_count();

    int num_min_inputs;
    int num_max_inputs;
    int num_min_outputs;
    int num_max_outputs;
};

class VectorElement : virtual public Element {
public:
    VectorElement() : Element() { }
    virtual int get_type() const { return ELEMTYPE_VECTOR; }

    int _process_batch(int input_port, PacketBatch *batch);

    /** User-defined vectorized packet processing function. */
    virtual int process_vector(int input_port, Packet **pkt_vec, vec_mask_arg_t mask) = 0;

    /** Unused. Should not be overriden. */
    int process(int input_port, Packet *pkt) { assert(0); return -1; }
};

class PerBatchElement : virtual public Element {
public:
    PerBatchElement() : Element() { }
    virtual int get_type() const { return ELEMTYPE_PER_BATCH; }

    int _process_batch(int input_port, PacketBatch *batch);

    /** User-defined function to process a batch. */
    virtual int process_batch(int input_port, PacketBatch *batch) = 0;

    /** Unused. Should not be overriden. */
    int process(int input_port, Packet *pkt) { assert(0); return -1; }
};

class SchedulableElement : virtual public Element {
public:
    SchedulableElement() : Element(), _last_call_ts(0), _last_delay(0), _last_check_tick(0) { }
    virtual int get_type() const { return ELEMTYPE_SCHEDULABLE; }

    /**
     * The user-defined body must return the output port number.
     * It may or may not return a PacketBatch.
     *
     * If out_batch is set to nullptr, the framework does not execute any
     * connected elements and ignores the return value.
     * Otherwise, the return value is treated as output_port index like
     * normal elements, and the framework executes next elements according
     * to it.
     *
     * Using both PerBatchElement and SchedulableElement can implement a Queue.
     * At its process_batch(), append the given batch pointer to an
     * internal vector.  At dispatch(), pop and return the batch pointer
     * from the internal vector.
     *
     * If next_delay value is set to larger than zero, it is interpreted as
     * a delay before the next scheduling, in microseconds.
     */
    virtual int dispatch(uint64_t loop_count, PacketBatch*& out_batch, uint64_t &next_delay) = 0;

    uint64_t _last_call_ts;
    uint64_t _last_delay;
    uint64_t _last_check_tick;
};

class OffloadableElement : virtual public SchedulableElement {

    friend class ElementGraph;

private:
    static const size_t MAX_FINBATCH_QLEN = NBA_MAX_COPROC_PPDEPTH * 16;

public:
    int reuse_head_ref = 0;
    int reuse_tail_ref = 0;

    OffloadableElement() : SchedulableElement()
    {
        if (dummy_device) {
            auto ch = [this] (ComputeContext *ctx, struct resource_param *res) {
                this->dummy_compute_handler(ctx, res);
            };
            offload_compute_handlers.insert({{"dummy", ch},});
        }
        finished_batches.init(MAX_FINBATCH_QLEN, -1, finished_batches_arrbuf);
        memset(tasks, 0, sizeof(OffloadTask *) * NBA_MAX_COPROCESSOR_TYPES);
    }
    virtual ~OffloadableElement() {}
    int get_type() const { return ELEMTYPE_OFFLOADABLE | ELEMTYPE_SCHEDULABLE; }

    /** Enqueues the given batch for offloading and begins offloading when
     * it has sufficient amount of work. */
    int offload(ElementGraph *mother, PacketBatch *in_batch, int input_port);

    /** Immediately begins offloading of the given (reused) offload-task. */
    int offload(ElementGraph *mother, OffloadTask *reused_offl_task, int input_port);

    /** Stores the batches that are returned from offloading. */
    int enqueue_batch(PacketBatch *batch);

    /** Resumes the element graph processing using the enqueued batches. */
    int dispatch(uint64_t loop_count, PacketBatch*& out_batch, uint64_t &next_delay);

    /** Returns the list of supported devices for offloading. */
    virtual void get_supported_devices(std::vector<std::string> &device_names) const = 0;
    //virtual size_t get_desired_workgroup_size(const char *device_name) const = 0;
    virtual int get_offload_item_counter_dbid() const = 0;
    virtual size_t get_used_datablocks(int *datablock_ids) = 0;
    //virtual void get_datablocks(std::vector<int> &datablock_ids){get_used_datablocks(datablock_ids);}; //TODO fill here...

    virtual int postproc(int input_port, void *custom_output, Packet *pkt) { return 0; }

    /** Offload handlers are executed in the coprocessor thread. */
    // TODO: optimize (avoid using unordered_map)?
    std::unordered_map<std::string, offload_compute_handler> offload_compute_handlers;
    std::unordered_map<std::string, offload_init_handler> offload_init_handlers;

private:
    OffloadTask *tasks[NBA_MAX_COPROCESSOR_TYPES];
    FixedRing<PacketBatch*, nullptr> finished_batches;
    PacketBatch *finished_batches_arrbuf[MAX_FINBATCH_QLEN];
    void dummy_compute_handler(ComputeContext *ctx, struct resource_param *res);
};

}

#endif

// vim: ts=8 sts=4 sw=4 et
