#ifndef __NBA_ELEMENT_DISCARD_HH__
#define __NBA_ELEMENT_DISCARD_HH__

#include <nba/element/element.hh>
#include <vector>
#include <string>

namespace nba {

class Discard : public Element {
public:
    Discard(): Element()
    {
    }

    ~Discard()
    {
    }

    const char *class_name() const { return "Discard"; }
    const char *port_count() const { return "1/1"; }

    int initialize();
    int initialize_global() { return 0; };      // per-system configuration
    int initialize_per_node() { return 0; };    // per-node configuration
    int configure(comp_thread_context *ctx, std::vector<std::string> &args);

    int process(int input_port, Packet *pkt);
};

EXPORT_ELEMENT(Discard);

}

#endif

// vim: ts=8 sts=4 sw=4 et
