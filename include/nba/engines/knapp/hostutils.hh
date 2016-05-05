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

constexpr uint16_t get_host_data_port(unsigned vdev_id)
{
    return KNAPP_HOST_DATA_PORT_BASE + vdev_id;
}

constexpr uint16_t get_host_ctrl_port(unsigned vdev_id)
{
    return KNAPP_HOST_CTRL_PORT_BASE + vdev_id;
}

constexpr uint16_t get_mic_data_port(unsigned vdev_id)
{
    return KNAPP_MIC_DATA_PORT_BASE + vdev_id;
}

constexpr uint16_t get_mic_ctrl_port(unsigned vdev_id)
{
    return KNAPP_MIC_CTRL_PORT_BASE + vdev_id;
}

void ctrl_invoke(scif_epd_t ep, const CtrlRequest &req, CtrlResponse &resp);

void connect_with_retry(struct vdevice *vdev);

int pollring_init(struct poll_ring *r, int32_t n, scif_epd_t epd, int node);

}} // endns(nba::knapp)


extern "C" {

/* Nothing here yet. */

}

#endif // __NBA_KNAPP_HOSTUTILS_HH__

// vim: ts=8 sts=4 sw=4 et
