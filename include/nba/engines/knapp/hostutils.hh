#ifndef __NBA_KNAPP_HOSTUTILS_HH__
#define __NBA_KNAPP_HOSTUTILS_HH__

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <scif.h>
#include <vector>
#include <nba/engines/knapp/defs.hh>
#include <nba/engines/knapp/hosttypes.hh>
#include <nba/engines/knapp/sharedtypes.hh>
#include <nba/engines/knapp/ctrl.pb.h>

namespace nba { namespace knapp {

extern std::vector<uint16_t> remote_scif_nodes;
extern uint16_t local_node;

void ctrl_invoke(scif_epd_t ep, const CtrlRequest &req, CtrlResponse &resp);

}} // endns(nba::knapp)


extern "C" {

/* Nothing here yet. */

}

#endif // __NBA_KNAPP_HOSTUTILS_HH__

// vim: ts=8 sts=4 sw=4 et
