#ifndef __NBA_ELEMENT_HH__
#define __NBA_ELEMENT_HH__

#include "types.hh"
extern "C" {
#include <rte_config.h>
#include <rte_memory.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
}
#include <cassert>
#include <string>
#include <vector>
#include <functional>
#include "config.hh"
#include "packet.hh"
#include "packetbatch.hh"
#include "offloadtask.hh"
#include "nodelocalstorage.hh"

namespace nba {

class Packet;   /* forward declaration */
class Element;  /* forward declaration */

enum ElementType {
    /* PER_PACKET and PER_BATCH are exclusive to each other. */
    ELEMTYPE_PER_PACKET = 1,
    ELEMTYPE_PER_BATCH = 2,
    ELEMTYPE_SCHEDULABLE = 4,
    ELEMTYPE_OFFLOADABLE = 8,
    ELEMTYPE_INPUT = 16,
    ELEMTYPE_OUTPUT = 32,
};

struct element_info {
    int idx;
    /* NOTE: Non-mutable lambda expression of the same type is
     * converted to the function pointer by the compiler. */
    Element* (*instantiate)(void);
};

#define EXPORT_ELEMENT(...)

class Element {
public:
    uint64_t branch_total = 0;
    uint64_t branch_miss = 0;
    uint64_t branch_count[16]; //XXX ElementGraph::num_max_outputs = 16

    Element();
    virtual ~Element();

    /* == User-defined properties and methods == */
    virtual const char *class_name() const = 0;
    virtual const char *port_count() const = 0;
    virtual int get_type() const { return ELEMTYPE_PER_PACKET; }

    virtual int configure(comp_thread_context *ctx, std::vector<std::string> &args);
    virtual int initialize() = 0;       // per-thread configuration. Called after coprocessor threads are initialized.
    virtual int initialize_global();    // thread-global configuration. Called before coprocessor threads are initialized.
    virtual int initialize_per_node();  // per-node configuration. Called before coprocessor threads are initialized.

    /** User-define function to process a packet. */
    virtual int process(int input_port,
                struct rte_mbuf *pkt,
                struct annotation_set *anno) = 0;

    /**
     * A framework-internal base method always called by the element graph.
     */
    virtual int _process_batch(int input_port, PacketBatch *batch);

    comp_thread_context *ctx;

protected:
    std::vector<Element*> next_elems;
    std::vector<int> next_connected_inputs;

    Packet *packet;

    // used to manage per-node info
    int num_nodes;
    int node_idx;

private:
    friend class ElementGraph;
    friend class Classifier;

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

class PerBatchElement : virtual public Element {
public:
    PerBatchElement() : Element() {}
    virtual int get_type() const { return ELEMTYPE_PER_BATCH; }

    int _process_batch(int input_port, PacketBatch *batch);

    /** User-defined function to process a batch. */
    virtual int process_batch(int input_port, PacketBatch *batch) = 0;

    /** Unused. Should not be overriden. */
    int process(int input_port, struct rte_mbuf *pkt, struct annotation_set *anno) { assert(0); return -1; }
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

class OffloadableElement : virtual public Element {
    friend class ElementGraph;
public:
    OffloadableElement() : Element()
    {
        if (dummy_device) {
            auto ch = [this] (ComputeContext *ctx, struct resource_param *res,
                              struct annotation_set **anno_ptr_array) {
                this->dummy_compute_handler(ctx, res, anno_ptr_array);
            };
            offload_compute_handlers.insert({{"dummy", ch},});
        }
        for (int i = 0; i < NBA_MAX_COPROCESSOR_TYPES; i++)
            tasks[i] = nullptr;
    }
    virtual ~OffloadableElement() {}
    int get_type() const { return ELEMTYPE_OFFLOADABLE; }

    virtual void get_supported_devices(std::vector<std::string> &device_names) const = 0;
    virtual void get_input_roi(struct input_roi_info *roi) const = 0;
    virtual void get_output_roi(struct output_roi_info *roi) const = 0;
    virtual size_t get_desired_workgroup_size(const char *device_name) const = 0;

    virtual void preproc(int input_port, void *custom_input,  struct rte_mbuf *pkt, struct annotation_set *anno) = 0;
    virtual void prepare_input(ComputeContext *ctx, struct resource_param *res, struct annotation_set **anno_ptr_array) = 0;
    virtual int postproc(int input_port, void *custom_output, struct rte_mbuf *pkt, struct annotation_set *anno) = 0;

    void preproc_batch(int input_port, void *custom_input, PacketBatch *batch);
    int *postproc_batch(int input_port, void *custom_output, PacketBatch *batch);

    /** Offload handlers are executed in the coprocessor thread. */
    // TODO: optimize (avoid using unordered_map)?
    std::unordered_map<std::string, offload_compute_handler> offload_compute_handlers;
    std::unordered_map<std::string, offload_init_handler> offload_init_handlers;

private:
    OffloadTask *tasks[NBA_MAX_COPROCESSOR_TYPES];
    void dummy_compute_handler(ComputeContext *ctx, struct resource_param *res, struct annotation_set **anno_ptr_array);
};

}

#endif

// vim: ts=8 sts=4 sw=4 et