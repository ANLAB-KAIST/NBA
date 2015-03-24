#ifndef __NBA_ELEMENT_ETHER_ARPQUERIER_HH__
#define __NBA_ELEMENT_ETHER_ARPQUERIER_HH__


#include <rte_config.h>
#include <rte_memory.h>
#include <rte_mbuf.h>
#include <rte_ether.h>

#include "../../lib/element.hh"
#include "../../lib/annotation.hh"
//#include "../../lib/nodelocalstorage.hh"
#include <vector>
#include <string>

#include "util_arptable.hh"

#include <net/if_arp.h>
#include <netinet/in.h>
#include <netinet/ip.h>

namespace nba {

class ARPQuerier : public SchedulableElement {
public:
    ARPQuerier(): SchedulableElement()
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

    int process(int input_port, struct rte_mbuf *pkt, struct annotation_set *anno);
    int dispatch(uint64_t loop_count, PacketBatch*& out_batch, uint64_t &next_delay);

private:
    unsigned capacity_pkt;
    unsigned capacity_arp_entry;
    unsigned timeout_arp_entry;
    unsigned renewal_timeout;

    struct timespec prev;

    static ARPTable *_table;    // per-element ARP table, which is malloced on heap. (Is it okay??)
};

EXPORT_ELEMENT(ARPQuerier);

}

#endif

// vim: ts=8 sts=4 sw=4 et
