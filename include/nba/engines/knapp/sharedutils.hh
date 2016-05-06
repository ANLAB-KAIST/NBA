#ifndef __NBA_KNAPP_SHAREDUTILS_HH__
#define __NBA_KNAPP_SHAREDUTILS_HH__

#include <nba/engines/knapp/defs.hh>

namespace nba { namespace knapp {

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


}} //endns(nba::knapp)

#endif //__NBA_KNAPP_SHAREDUTILS_HH__

// vim: ts=8 sts=4 sw=4 et
