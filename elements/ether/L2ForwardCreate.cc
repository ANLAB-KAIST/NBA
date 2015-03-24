#include "L2ForwardCreate.hh"
#include "../../lib/types.hh"
#include <cassert>

#include <rte_string_fns.h>



using namespace std;
using namespace nshader;

int L2ForwardCreate::initialize()
{
    return 0;
}

int L2ForwardCreate::configure(comp_thread_context *ctx, std::vector<std::string> &args)
{
    Element::configure(ctx, args);
    mode = 0;
    return 0;
}

int L2ForwardCreate::process(int input_port, struct rte_mbuf *pkt, struct annotation_set *anno)
{
    struct ether_hdr *ethh = rte_pktmbuf_mtod(pkt, struct ether_hdr *);
    if (likely(is_unicast_ether_addr(&ethh->d_addr))) {

        unsigned iface_in = anno_get(anno, NSHADER_ANNO_IFACE_IN);
        struct ether_hdr* header = rte_pktmbuf_mtod(pkt, struct ether_hdr*);
        struct ether_addr temp;
        header->s_addr = this->ctx->io_ctx->tx_ports[iface_in].addr;
        header->d_addr = header->s_addr;

        this->ctx->io_tx_new(header, rte_pktmbuf_data_len(pkt), iface_in);

    }
    return DROP;
}

// vim: ts=8 sts=4 sw=4 et
