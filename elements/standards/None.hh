#ifndef __NBA_ELEMENT_NONE_HH__
#define __NBA_ELEMENT_NONE_HH__


#include <rte_config.h>
#include <rte_memory.h>
#include <rte_mbuf.h>
#include <rte_ether.h>

#include "../../lib/element.hh"
#include "../../lib/annotation.hh"
#include <vector>
#include <string>

namespace nba {

class None : public Element {
public:
    None(): Element()
    {
    }

    ~None()
    {
    }

    const char *class_name() const { return "None"; }
    const char *port_count() const { return "1/1"; }

    int initialize();
    int initialize_global() { return 0; };      // per-system configuration
    int initialize_per_node() { return 0; };    // per-node configuration
    int configure(comp_thread_context *ctx, std::vector<std::string> &args);

    int process(int input_port, struct rte_mbuf *pkt, struct annotation_set *anno);
};

EXPORT_ELEMENT(None);

}

#endif

// vim: ts=8 sts=4 sw=4 et
