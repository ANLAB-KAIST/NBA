#ifndef __NBA_ELEMENT_ETHER_L2FWD_HH__
#define __NBA_ELEMENT_ETHER_L2FWD_HH_

extern "C" {
#include <rte_config.h>
#include <rte_memory.h>
#include <rte_mbuf.h>
#include <rte_ether.h>
}
#include "../../lib/element.hh"
#include "../../lib/annotation.hh"
#include <vector>
#include <string>

namespace nba {

class L2Forward : public Element {
public:
    L2Forward(): Element()
    {
    }

    ~L2Forward()
    {
    }

    const char *class_name() const { return "L2ForwardElement"; }
    const char *port_count() const { return "1/1"; }

    int initialize() { return 0; }
    int initialize_global() { return 0; }      // per-system configuration
    int initialize_per_node() { return 0; }    // per-node configuration
    int configure(comp_thread_context *ctx, std::vector<std::string> &args);

    int process(int input_port, struct rte_mbuf *pkt, struct annotation_set *anno);
private:
    int mode;
    unsigned next_port;
    uint64_t last_batch_id;
};

EXPORT_ELEMENT(L2Forward);

}

#endif

// vim: ts=8 sts=4 sw=4 et