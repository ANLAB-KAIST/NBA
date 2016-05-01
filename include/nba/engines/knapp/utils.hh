#ifndef __NBA_KNAPP_UTILS_HH__
#define __NBA_KNAPP_UTILS_HH__

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <scif.h>
#include <vector>
#include <nba/engines/knapp/defs.hh>
#include <nba/engines/knapp/types.hh>

namespace nba {
namespace knapp {

extern std::vector<uint16_t> remote_scif_nodes;
extern uint16_t local_node;

constexpr uint16_t get_host_dataport(unsigned vdev_id)
{
    return KNAPP_HOST_DATAPORT_BASE + vdev_id;
}

constexpr uint16_t get_host_ctrlport(unsigned vdev_id)
{
    return KNAPP_HOST_CTRLPORT_BASE + vdev_id;
}

constexpr uint16_t get_mic_dataport(unsigned vdev_id)
{
    return KNAPP_MIC_DATAPORT_BASE + vdev_id;
}

constexpr uint16_t get_mic_ctrlport(unsigned vdev_id)
{
    return KNAPP_MIC_CTRLPORT_BASE + vdev_id;
}

void connect_with_retry(struct vdevice *vdev);

} // endns(knapp)
} // endns(nba)


extern "C" {

/* Nothing here yet. */

}

#endif // __KNAPP_UTILS_HH__

// vim: ts=8 sts=4 sw=4 et
