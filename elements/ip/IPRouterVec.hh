#ifndef __NBA_ELEMENT_IP_IPROUTERVEC_HH__
#define __NBA_ELEMENT_IP_IPROUTERVEC_HH__

#include <nba/element/element.hh>
#include <vector>
#include <string>

namespace nba {

class IPRouterVec : public VectorElement {
public:
    IPRouterVec(): VectorElement()
    {
    }

    virtual ~IPRouterVec()
    {
    }

    const char *class_name() const { return "IPRouterVec"; }
    const char *port_count() const { return "1/1"; }

    int initialize() { return 0; }
    int initialize_global() { return 0; };      // per-system configuration
    int initialize_per_node() { return 0; };    // per-node configuration
    int configure(comp_thread_context *ctx, std::vector<std::string> &args)
    { return 0; }

    int process_vector(int input_port, Packet **pkt_vec, vec_mask_arg_t mask);
};

EXPORT_ELEMENT(IPRouterVec);

}

#endif

// vim: ts=8 sts=4 sw=4 et
