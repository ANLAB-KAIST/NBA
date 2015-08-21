#ifndef __NBA_ELEMENT_ETHER_ARPQUERIER_HH__
#define __NBA_ELEMENT_ETHER_ARPQUERIER_HH__

#include <nba/element/element.hh>
#include <vector>
#include <string>

namespace nba {

class ARPTable;

class ARPQuerier : public SchedulableElement {
public:
    ARPQuerier(): SchedulableElement(), _table(NULL)
    {
        prev = {0, 0};
    }

    ~ARPQuerier()
    {
    }

    const char *class_name() const { return "ARPQuerier"; }
    const char *port_count() const { return "2/1"; }

    int initialize();
    int initialize_global();        // per-system configuration
    int initialize_per_node();  // per-node configuration
    int configure(comp_thread_context *ctx, std::vector<std::string> &args);

    int process(int input_port, Packet *pkt);
    int dispatch(uint64_t loop_count, PacketBatch*& out_batch, uint64_t &next_delay);

private:
    unsigned capacity_pkt;
    unsigned capacity_arp_entry;
    unsigned timeout_arp_entry;
    unsigned renewal_timeout;

    struct timespec prev;

    ARPTable *_table;  // FIXME: use node-local storage
};

EXPORT_ELEMENT(ARPQuerier);

}

#endif

// vim: ts=8 sts=4 sw=4 et
