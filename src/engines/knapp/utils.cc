#include <nba/engines/knapp/types.hh>
#include <nba/engines/knapp/utils.hh>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <unistd.h>
#include <scif.h>
#include <rte_common.h>

using namespace nba::knapp;

/* Detected in the configuration loading phase. (config.cc) */
std::vector<uint16_t> nba::knapp::remote_scif_nodes;
uint16_t nba::knapp::local_node;


void nba::knapp::connect_with_retry(struct vdevice *vdev)
{
    int rc = 0;
    fprintf(stderr, "connect_with_retry\n");
    for (unsigned retry = 0; retry < KNAPP_SCIF_MAX_CONN_RETRY; retry++) {
        rc = scif_connect(vdev->data_epd, &vdev->remote_dataport);
        if (rc < 0) {
            fprintf(stderr, "vdevice %d could not connect to remote data port (%d, %d). Retrying (%u) ...\n",
                    vdev->device_id, vdev->remote_dataport.node, vdev->remote_dataport.port, retry + 1);
            usleep(500000);
            continue;
        }
        fprintf(stderr, "vdevice %d connected to remote data port (%d, %d).\n",
                vdev->device_id, vdev->remote_dataport.node, vdev->remote_dataport.port);
        break;
    }
    if (rc < 0) {
        fprintf(stderr, "Failed to connect vdevice %d to remote data port (%d, %d). Error code %d\n",
                vdev->device_id, vdev->remote_dataport.node, vdev->remote_dataport.port, rc);
        rte_exit(EXIT_FAILURE, "All 5 data port connection attemps have failed\n");
    }
    for (unsigned retry = 0; retry < KNAPP_SCIF_MAX_CONN_RETRY; retry++) {
        rc = scif_connect(vdev->ctrl_epd, &vdev->remote_ctrlport);
        if (rc < 0) {
            fprintf(stderr, "vdevice %d could not connect to remote control port (%d, %d). Retrying (%u) ...\n",
                    vdev->device_id, vdev->remote_ctrlport.node, vdev->remote_ctrlport.port, retry + 1);
            usleep(500000);
            continue;
        }
        fprintf(stderr, "vdevice %d connected to remote control port (%d, %d).\n",
                vdev->device_id, vdev->remote_ctrlport.node, vdev->remote_ctrlport.port);
        break;
    }
    if (rc < 0) {
        fprintf(stderr, "Failed to connect vdevice %d to remote control port (%d, %d). Error code %d\n",
                vdev->device_id, vdev->remote_ctrlport.node, vdev->remote_ctrlport.port, rc);
        rte_exit(EXIT_FAILURE, "All 5 control port connection attemps have failed\n");
    }
}

// vim: ts=8 sts=4 sw=4 et
