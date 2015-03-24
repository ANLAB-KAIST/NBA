#include "CheckIPHeader.hh"
#include "../../lib/types.hh"
#include "../../lib/log.hh"
#include "../util/util_checksum.hh"

using namespace std;
using namespace nshader;

int CheckIPHeader::initialize()
{
    // TODO: Depending on configuration,
    //       make a rx-to-tx address mapping or perform echo-back.
    //       The current default is echo-back.
    return 0;
}

int CheckIPHeader::configure(comp_thread_context *ctx, std::vector<std::string> &args)
{
    Element::configure(ctx, args);
    return 0;
}

int CheckIPHeader::process(int input_port, struct rte_mbuf *pkt, struct annotation_set *anno)
{
    struct ether_hdr *ethh = rte_pktmbuf_mtod(pkt, struct ether_hdr *);
    struct iphdr *iph = (struct iphdr *)(ethh + 1);

    if (ntohs(ethh->ether_type) != ETHER_TYPE_IPv4) {
        RTE_LOG(DEBUG, ELEM, "CheckIPHeader: invalid packet type - %x\n", ntohs(ethh->ether_type));
        return DROP;
    }

    if ( (iph->version != 4) || (iph->ihl < 5) ) {
        RTE_LOG(DEBUG, ELEM, "CheckIPHeader: invalid packet - ver %d, ihl %d\n", iph->version, iph->ihl);
        return SLOWPATH;
    }

    if ( (iph->ihl * 4) > ntohs(iph->tot_len)) {
        RTE_LOG(DEBUG, ELEM, "CheckIPHeader: invalid packet - total len %d, ihl %d\n", iph->tot_len, iph->ihl);
        return SLOWPATH;
    }

    // TODO: Discard illegal source addresses.

    if (ip_fast_csum(iph, iph->ihl) != 0) {
        return DROP;
    }

    return 0; // output port number: 0
}

// vim: ts=8 sts=4 sw=4 et
