#include "IPRouterVec.hh"
#include <nba/core/vector.hh>
#include <nba/element/element.hh>

using namespace nba;

int IPRouterVec::process_vector(int input_port,
                                Packet **pkt_vec,
                                vec_mask_arg_t mask)
{
    // A temporary scalar no-op implementation.
    for (int i = 0; i < NBA_VECTOR_WIDTH; i++)
        if (mask.m[i])
            output(0).push(pkt_vec[i]);
    return 0; // ignored
}

// vim: ts=8 sts=4 sw=4 et
