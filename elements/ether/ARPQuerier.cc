#include <nba/core/intrinsic.hh>
#include <nba/element/element.hh>
#include <nba/element/packet.hh>
#include "ARPQuerier.hh"
#include "util_arptable.hh"
#include <net/if_arp.h>
#include <netinet/in.h>
#include <netinet/ip.h>

using namespace std;
using namespace nba;

int ARPQuerier::initialize()
{
    return 0;
}

// per-system configuration
int ARPQuerier::initialize_global()
{
    _table->configure(capacity_pkt, capacity_arp_entry, timeout_arp_entry);
    return 0;
};

// per-node configuration
int ARPQuerier::initialize_per_node() {
    return 0;
};

int ARPQuerier::configure(comp_thread_context *ctx, std::vector<std::string> &args)
{
    Element::configure(ctx, args);

    // Define IP packet capacity, ARP entry capacity, ARP entry timeout, Poll time gap, Broadcast addr, Broadcast poll
    // Default value: 2048, 0(unlimit), 5 min, 1 min, ?, false

    capacity_pkt = 2048;
    capacity_arp_entry = 0;
    timeout_arp_entry = 5;
    renewal_timeout = 1;

    _table = NULL;

    return 0;
}

int ARPQuerier::process(int input_port, Packet *pkt)
{
    struct ether_hdr *ethh = (struct ether_hdr *) pkt->data();

    if (input_port == 0) {
        struct ether_arp *arp_pkt = (struct ether_arp *)(ethh + 1); // ARP header & src/dst addrs
        struct arphdr *arph = &arp_pkt->ea_hdr;

        // ARPResponse packet comes in..
        if ( (ntohs(ethh->ether_type) != ETHER_TYPE_ARP)
                && (ntohs(arph->ar_op) != ARPOP_REPLY)
                && (ntohs(arph->ar_hrd) != ARPHRD_ETHER)
                && (ntohs(arph->ar_pro) != ETHER_TYPE_IPv4) ) {
                return DROP;
        }

        uint32_t new_ip_addr;
        EtherAddress new_eth_addr;

        //new_ip_addr = *((uint32_t*)arp_pkt->arp_spa);
        memcpy(&new_ip_addr, arp_pkt->arp_spa, sizeof(uint32_t));
        new_ip_addr = ntohl(new_ip_addr);
        new_eth_addr.set(arp_pkt->arp_sha);

        _table->insert(new_ip_addr, (const EtherAddress)new_eth_addr);

        // TODO: iterate ARP entry in ARP table & send pkts with added dest ip addr.
    }
    else {
        // Find matching mac addr for forwarded packets..
        // Assumes IPv4 packet.
        if (ntohs(ethh->ether_type) != ETHER_TYPE_IPv4) {
                return DROP;
        }

        struct iphdr *iph = (struct iphdr *)(ethh + 1);
        EtherAddress dest_addr;

        dest_addr = _table->lookup(ntohl(iph->daddr));
        if (!dest_addr.is_broadcast()) {
            // Matching mac addr found. revise dest mac addr & forward packet.
            int i;
            for (i=0; i<6; ++i) {
                ethh->d_addr.addr_bytes[i] = dest_addr._data[i];
            }
        }
        else {
            // TODO: If not found, pkt should be stored to ARP table
            // & ARP request pkt should be generated & echo-backed to input port.

            // return (what?) as whole pkt is stored to ARP table, drop it?
            return PENDING;
        }
    }

    return 0;
}

int ARPQuerier::dispatch(uint64_t loop_count, PacketBatch*& out_batch, uint64_t &next_delay)
{
    _table->handle_timer();
    next_delay = 1000000; // 1 sec (in us)
    out_batch = nullptr;
    return 0;
}

// vim: ts=8 sts=4 sw=4 et
