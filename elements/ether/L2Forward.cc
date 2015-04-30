#include "L2Forward.hh"
#include "../../lib/types.hh"
#include <cassert>

#include <rte_string_fns.h>


using namespace std;
using namespace nba;

const thread_local int ECHOBACK_PAIRED = 0;
const thread_local int ECHOBACK_NUMA_CROSS = 1;
const thread_local int RR_PER_PACKET = 2;
const thread_local int RR_PER_BATCH = 3;
const thread_local int TWO_TO_ONE = 4;

int L2Forward::configure(comp_thread_context *ctx, std::vector<std::string> &args)
{
    Element::configure(ctx, args);
    mode = 0;
    char *split_args[3];
    char arg_copy[128];
    if (args.size() == 1) {
        strcpy(arg_copy, args[0].c_str());
        rte_strsplit(arg_copy, args[0].size(), split_args, 3, ' ');
        if (!strcmp(split_args[0], "method")) {
            if (!strcmp(split_args[1], "echoback")) {
                mode = ECHOBACK_PAIRED;
            } else if (!strcmp(split_args[1], "echoback_cross")) {
                mode = ECHOBACK_NUMA_CROSS;
            } else if (!strcmp(split_args[1], "two_to_one")) {
                mode = TWO_TO_ONE;
            } else if (!strcmp(split_args[1], "roundrobin")) {
                mode = RR_PER_PACKET;
                next_port = 0;
            } else if (!strcmp(split_args[1], "roundrobin_batch")) {
                mode = RR_PER_BATCH;
                next_port = 0;
                last_batch_id = 0;
            }
        }
    } else
        assert(0); // Not enough number of configuration arguments!
    return 0;
}

int L2Forward::process(int input_port, Packet *pkt)
{
    struct ether_hdr *ethh = (struct ether_hdr *) pkt->data();
    if (likely(is_unicast_ether_addr(&ethh->d_addr))) {
        switch(mode) {
          case ECHOBACK_PAIRED:
          {
            unsigned iface_in = anno_get(&pkt->anno, NBA_ANNO_IFACE_IN);
            unsigned iface_out = iface_in + ((iface_in % 2) ? -1 : +1);
            anno_set(&pkt->anno, NBA_ANNO_IFACE_OUT, iface_out);
            break;
          }
          case ECHOBACK_NUMA_CROSS:
          {
            unsigned iface_in = anno_get(&pkt->anno, NBA_ANNO_IFACE_IN);
            unsigned iface_out = (iface_in + ctx->num_tx_ports/ctx->num_nodes) % (ctx->num_tx_ports);
            anno_set(&pkt->anno, NBA_ANNO_IFACE_OUT, iface_out);
            break;
          }
          case RR_PER_PACKET:
          {
            next_port = (next_port + 1) % (ctx->num_tx_ports);
            anno_set(&pkt->anno, NBA_ANNO_IFACE_OUT, next_port);
            break;
          }
          case RR_PER_BATCH:
          {
            uint64_t batch_id = anno_get(&pkt->anno, NBA_ANNO_BATCH_ID);
            if (last_batch_id != batch_id)
                next_port = (next_port + 1) % (ctx->num_tx_ports);
            anno_set(&pkt->anno, NBA_ANNO_IFACE_OUT, next_port);
            last_batch_id = batch_id;
            break;
          }
          case TWO_TO_ONE:
          {
            unsigned iface_in = anno_get(&pkt->anno, NBA_ANNO_IFACE_IN);
            unsigned iface_out = iface_in - (iface_in % 2);
            anno_set(&pkt->anno, NBA_ANNO_IFACE_OUT, iface_out);
            break;
          }
        }
        return 0;
    }
    return DROP;
}

// vim: ts=8 sts=4 sw=4 et
