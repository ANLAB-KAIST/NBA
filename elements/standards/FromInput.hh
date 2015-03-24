#ifndef __NSHADER_ELEMENT_FROMINPUT_HH__
#define __NSHADER_ELEMENT_FROMINPUT_HH__


#include <rte_config.h>
#include <rte_memory.h>
#include <rte_mbuf.h>

#include "../../lib/element.hh"
#include "../../lib/annotation.hh"
#include <vector>
#include <string>

namespace nshader {

class FromInput : public SchedulableElement, PerBatchElement {
    /**
     * FromInput element is a NULL element, which does nothing.
     * It serves as a placeholder in configuration to indicate
     * connections to the RX Common Component.
     */
public:
    FromInput() : SchedulableElement(), PerBatchElement()
    {
    }

    ~FromInput()
    {
    }

    const char *class_name() const { return "FromInput"; }
    const char *port_count() const { return "0/1"; }
    int get_type() const { return ELEMTYPE_INPUT | SchedulableElement::get_type() | PerBatchElement::get_type(); }

    int initialize() { return 0; };
    int initialize_global() { return 0; };
    int initialize_per_node() { return 0; };

    int configure(comp_thread_context *ctx, std::vector<std::string> &args)
    {
        Element::configure(ctx, args);
        return 0;
    }

    int process_batch(int input_port, PacketBatch *batch)
    {
        /* Always pass the batch to the next element. */
        return 0;
    }

    int dispatch(uint64_t loop_count, PacketBatch*& out_batch, uint64_t &next_delay)
    {
        /* "Generate" a packet batch, which is actually taken from the framework's RX part. */
        out_batch = ctx->input_batch;
        assert(out_batch != nullptr);
        next_delay = 0;
        return 0;
    }
};

EXPORT_ELEMENT(FromInput);

}

#endif

// vim: ts=8 sts=4 sw=4 et
