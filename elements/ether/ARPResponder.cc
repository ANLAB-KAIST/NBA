#include "ARPResponder.hh"
#include "../../lib/types.hh"
#include "sstream"

using namespace std;
using namespace nba;

int ARPResponder::initialize()
{
    return 0;
}

// per-system configuration
int ARPResponder::initialize_global()
{
    // Put IP addr - Mac addr mapping into hashmap.
    std::vector<std::string>::iterator it;
    string ip_addr_str, eth_addr_str;
    uint32_t ip_addr_num;
    EtherAddress eth_addr_obj;

    for (it = _args.begin(); it != _args.end(); ++it) {
        std::string str = *it;
        stringstream stream(str);
        stream >> ip_addr_str;
        stream >> eth_addr_str;

        //printf("ip_addr:%s, eth_addr:%s\n", ip_addr.c_str(), eth_addr.c_str());
        // TODO: It could be IP addr/mask, but for now, only IP addr is handled. (mask not yet).
        ip_addr_num = inet_addr(ip_addr_str.c_str());
        eth_addr_obj = EtherAddress(eth_addr_str);

         std::pair<uint32_t, EtherAddress> _pair (ip_addr_num, eth_addr_obj);
        _addr_hashmap.insert(_pair);
    }

    return 0;
};

// per-node configuration
int ARPResponder::initialize_per_node() {
    return 0;
};

int ARPResponder::configure(comp_thread_context *ctx, std::vector<std::string> &args)
{
    _args = args;

    return 0;
}

int ARPResponder::process(int input_port, struct rte_mbuf *pkt, struct annotation_set *anno)
{
    struct ether_hdr *ethh = rte_pktmbuf_mtod(pkt, struct ether_hdr *);

    struct ether_arp *arp_pkt = (struct ether_arp *)(ethh + 1); // ARP header & src/dst addrs
    struct arphdr *arph = &arp_pkt->ea_hdr;

    // ARP request packet comes in..
    if ( (ntohs(ethh->ether_type) != ETHER_TYPE_ARP)
            && (ntohs(arph->ar_op) != ARPOP_REQUEST)
            && (ntohs(arph->ar_hrd) != ARPHRD_ETHER)
            && (ntohs(arph->ar_pro) != ETHER_TYPE_IPv4) ) {
            return DROP;
    }

    uint32_t sender_ip_addr;
    EtherAddress sender_eth_addr;
    uint32_t dest_ip_addr;

    //sender_ip_addr = ntohl(*(uint32_t*)arp_pkt->arp_spa);
    memcpy(&sender_ip_addr, arp_pkt->arp_spa, sizeof(uint32_t));
    sender_eth_addr.set(arp_pkt->arp_sha);
    //dest_ip_addr = *(uint32_t*)arp_pkt->arp_tpa;
    memcpy(&dest_ip_addr, arp_pkt->arp_tpa, sizeof(uint32_t));

    unordered_map<uint32_t, EtherAddress>::iterator iter;
    iter = _addr_hashmap.find(dest_ip_addr);
    if (iter != _addr_hashmap.end()) {
        // Coresponding MAC address found on this system.
        assert(0);
        EtherAddress found_eth_addr = iter->second;

        // Create ARP reply packet.
        char *arp_rep_pkt = 0;
        struct ether_hdr *arp_rep_ethh = (struct ether_hdr *)arp_rep_pkt;
        found_eth_addr.put_to(&arp_rep_ethh->s_addr);   // reply eth src addr: addr found from hash_map
        sender_eth_addr.put_to(&arp_rep_ethh->d_addr);  // reply eth dest addt: addr from request pkt
        arp_rep_ethh->ether_type = ETHER_TYPE_ARP;

        struct ether_arp *arp_rep_payload = (struct ether_arp*) (arp_rep_pkt + sizeof(ether_hdr));
        memcpy(&arp_rep_payload->ea_hdr, arph, sizeof(arphdr));
        arp_rep_payload->ea_hdr.ar_op = ARPOP_REPLY;

        memcpy(arp_rep_payload->arp_sha, found_eth_addr._data, sizeof(uint8_t)*6);  // reply src mac addr: addr found from hashmap
        convert_ip_addr(dest_ip_addr, arp_rep_payload->arp_spa);                    // reply src ip addr: dest addr from request pkt
        memcpy(arp_rep_payload->arp_tha, sender_eth_addr._data, sizeof(uint8_t)*6); // reply dest mac addr: sender addr from request pkt
        convert_ip_addr(sender_ip_addr, arp_rep_payload->arp_tpa);                  // reply dest ip addr: sender addr from request pkt

        // Create prepacket struct

        // Make reply packet go to the port that request packt came in.

        // Put prepacket to prepacket_queue of io_thread_context
    }
    else {
        // not found, do nothing & drop it. (This is router!)
    }

    return DROP;
}

// vim: ts=8 sts=4 sw=4 et
