#include <nba/engines/knapp/hosttypes.hh>
#include <nba/engines/knapp/sharedtypes.hh>
#include <nba/engines/knapp/hostutils.hh>
#include <nba/engines/knapp/ctrl.pb.h>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <unistd.h>
#include <scif.h>
#include <rte_common.h>

using namespace nba::knapp;

/* Detected in the configuration loading phase. (config.cc) */
std::vector<uint16_t> nba::knapp::remote_scif_nodes;
uint16_t nba::knapp::local_node;


void nba::knapp::ctrl_invoke(scif_epd_t ep, const CtrlRequest &req, CtrlResponse &resp)
{
    int rc;
    char buf[1024];
    uint32_t msgsz = req.ByteSize();
    assert(msgsz < 1024u);
    req.SerializeToArray(buf, msgsz);
    rc = scif_send(ep, &msgsz, sizeof(msgsz), SCIF_SEND_BLOCK);
    assert(sizeof(msgsz) == rc);
    rc = scif_send(ep, buf, msgsz, SCIF_SEND_BLOCK);
    assert(msgsz == (unsigned) rc);
    resp.Clear();
    rc = scif_recv(ep, &msgsz, sizeof(msgsz), SCIF_RECV_BLOCK);
    assert(sizeof(msgsz) == rc);
    assert(msgsz < 1024u);
    rc = scif_recv(ep, buf, msgsz, SCIF_RECV_BLOCK);
    assert(msgsz == (unsigned) rc);
    resp.ParseFromArray(buf, msgsz);
}

void nba::knapp::connect_with_retry(struct vdevice *vdev)
{
    int rc = 0;
    fprintf(stderr, "connect_with_retry\n");
    for (unsigned retry = 0; retry < KNAPP_SCIF_MAX_CONN_RETRY; retry++) {
        rc = scif_connect(vdev->data_epd, &vdev->remote_data_port);
        if (rc < 0) {
            fprintf(stderr, "vdevice %d could not connect to remote data port (%d, %d). Retrying (%u) ...\n",
                    vdev->device_id, vdev->remote_data_port.node, vdev->remote_data_port.port, retry + 1);
            usleep(500000);
            continue;
        }
        fprintf(stderr, "vdevice %d connected to remote data port (%d, %d).\n",
                vdev->device_id, vdev->remote_data_port.node, vdev->remote_data_port.port);
        break;
    }
    if (rc < 0) {
        fprintf(stderr, "Failed to connect vdevice %d to remote data port (%d, %d). Error code %d\n",
                vdev->device_id, vdev->remote_data_port.node, vdev->remote_data_port.port, rc);
        rte_exit(EXIT_FAILURE, "All 5 data port connection attemps have failed\n");
    }
    for (unsigned retry = 0; retry < KNAPP_SCIF_MAX_CONN_RETRY; retry++) {
        rc = scif_connect(vdev->ctrl_epd, &vdev->remote_ctrl_port);
        if (rc < 0) {
            fprintf(stderr, "vdevice %d could not connect to remote control port (%d, %d). Retrying (%u) ...\n",
                    vdev->device_id, vdev->remote_ctrl_port.node, vdev->remote_ctrl_port.port, retry + 1);
            usleep(500000);
            continue;
        }
        fprintf(stderr, "vdevice %d connected to remote control port (%d, %d).\n",
                vdev->device_id, vdev->remote_ctrl_port.node, vdev->remote_ctrl_port.port);
        break;
    }
    if (rc < 0) {
        fprintf(stderr, "Failed to connect vdevice %d to remote control port (%d, %d). Error code %d\n",
                vdev->device_id, vdev->remote_ctrl_port.node, vdev->remote_ctrl_port.port, rc);
        rte_exit(EXIT_FAILURE, "All 5 control port connection attemps have failed\n");
    }
}


// vim: ts=8 sts=4 sw=4 et
