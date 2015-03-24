#ifndef __NBA_ELEMENT_LOADBALANCEBYENV_HH__
#define __NBA_ELEMENT_LOADBALANCEBYENV_HH__

#include "../../lib/element.hh"
#include "../../lib/annotation.hh"

#include <rte_errno.h>
#include <rte_log.h>

#include <vector>
#include <string>

namespace nba {

class LoadBalanceByEnv : public SchedulableElement, PerBatchElement {
public:
    LoadBalanceByEnv() : SchedulableElement(), PerBatchElement(), lb_decision(-1)
    { }

    virtual ~LoadBalanceByEnv()
    { }

    const char *class_name() const { return "LoadBalanceByEnv"; }
    const char *port_count() const { return "1/1"; }
    int get_type() const { return SchedulableElement::get_type() | PerBatchElement::get_type(); }

    int initialize() { return 0; }
    int initialize_global() { return 0; }
    int initialize_per_node() { return 0; }

    int configure(comp_thread_context *ctx, std::vector<std::string> &args)
    {
        Element::configure(ctx, args);
        char *lb_mode = getenv("NBA_LOADBALANCER_MODE");
        if (lb_mode == nullptr)
            lb_mode = const_cast<char*>("CPUOnly");

        if (!strcmp(lb_mode, "CPUOnly")) {
            lb_decision = -1;
        } else if (!strcmp(lb_mode, "GPUOnly")) {
            lb_decision = 0;
        } else {
            rte_panic("Unsupported load balancer mode: %s\n", lb_mode);
        }
        RTE_LOG(INFO, LB, "load balancer mode: %s\n", lb_mode);
        return 0;
    }

    int process_batch(int input_port, PacketBatch *batch)
    {
        anno_set(&batch->banno, NBA_BANNO_LB_DECISION, lb_decision);
        return 0;
    }

    int dispatch(uint64_t loop_count, PacketBatch*& out_batch, uint64_t &next_delay)
    {
        next_delay = 1e6L;
        out_batch = nullptr;
        return 0;
    }

private:
    int64_t lb_decision;
};

EXPORT_ELEMENT(LoadBalanceByEnv);

}

#endif

// vim: ts=8 sts=4 sw=4 et
