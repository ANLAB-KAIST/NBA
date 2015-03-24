#ifndef __NSHADER_ELEMENT_QUEUE_HH__
#define __NSHADER_ELEMENT_QUEUE_HH__


#include <rte_errno.h>

#include "../../lib/element.hh"
#include "../../lib/annotation.hh"
#include "../../lib/queue.hh"
#include <vector>
#include <string>

namespace nshader {

class Queue : public SchedulableElement, PerBatchElement {
public:
    Queue() : SchedulableElement(), PerBatchElement()
    {
        max_size = 0;
    }

    ~Queue()
    {
        delete queue;
    }

    const char *class_name() const { return "Queue"; }
    const char *port_count() const { return "1/1"; }
    int get_type() const { return SchedulableElement::get_type() | PerBatchElement::get_type(); }

    int initialize() {
        queue = new FixedRing<PacketBatch*, nullptr>(max_size, ctx->loc.node_id);
        return 0;
    };
    int initialize_global() { return 0; };
    int initialize_per_node() { return 0; };

    int configure(comp_thread_context *ctx, std::vector<std::string> &args)
    {
        Element::configure(ctx, args);
        // TODO: read max_size
        max_size = 512;
        return 0;
    }

    int process_batch(int input_port, PacketBatch *batch)
    {
        if (queue->size() == max_size)
            rte_panic("Queue overflow!\n");
        queue->push_back(batch);
        return KEPT_BY_ELEMENT;
    }

    int dispatch(uint64_t loop_count, PacketBatch*& out_batch, uint64_t &next_delay)
    {
        if (queue->size() > 0) {
            out_batch = queue->front();
            queue->pop_front();
        } else
            out_batch = nullptr;
        next_delay = 0;
        return 0;
    }

private:
    size_t max_size;
    FixedRing<PacketBatch*, nullptr> *queue;
};

EXPORT_ELEMENT(Queue);

}

#endif

// vim: ts=8 sts=4 sw=4 et
