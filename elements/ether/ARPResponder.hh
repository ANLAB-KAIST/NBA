#ifndef __NSHADER_ELEMENT_ETHER_ARPRESPONDER_HH__
#define __NSHADER_ELEMENT_ETHER_ARPRESPONDER_HH__


#include <rte_config.h>
#include <rte_memory.h>
#include <rte_mbuf.h>
#include <rte_ether.h>

#include "../../lib/element.hh"
#include "../../lib/annotation.hh"
//#include "../../lib/nodelocalstorage.hh"
#include <vector>
#include <string>
#include <unordered_map>

#include <netinet/in.h>
#include <netinet/ip.h>
#include <arpa/inet.h>

#include "util_arptable.hh"

namespace nshader {

class ARPResponder : public Element {
public:
    ARPResponder(): Element()
    {
    }

    ~ARPResponder()
    {
    }

    const char *class_name() const { return "ARPResponder"; }
    const char *port_count() const { return "1/1"; }

    int initialize();
    int initialize_global();        // per-system configuration
    int initialize_per_node();  // per-node configuration
    int configure(comp_thread_context *ctx, std::vector<std::string> &args);

    int process(int input_port, struct rte_mbuf *pkt, struct annotation_set *anno);

private:
    std::vector<std::string> _args;
    std::unordered_map<uint32_t, EtherAddress> _addr_hashmap;
};

EXPORT_ELEMENT(ARPResponder);

}

#endif

// vim: ts=8 sts=4 sw=4 et
