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

static int _global_pollring_counter = 0;


void nba::knapp::ctrl_invoke(scif_epd_t ep, const CtrlRequest &req, CtrlResponse &resp)
{
    char buf[1024];
    uint32_t msgsz = req.ByteSize();
    assert(msgsz < 1024);
    req.SerializeToArray(buf, msgsz);
    rc = scif_send(ep, &msgsz, sizeof(msgsz), SCIF_SEND_BLOCK);
    assert(sizeof(msgsz) == rc);
    rc = scif_send(ep, buf, msgsz, SCIF_SEND_BLOCK);
    assert(msgsz == rc);
    resp.Clear();
    rc = scif_recv(ep, &msgsz, sizeof(msgsz), SCIF_RECV_BLOCK);
    assert(sizeof(msgsz) == rc);
    assert(msgsz < 1024);
    rc = scif_recv(ep, buf, msgsz, SCIF_RECV_BLOCK);
    assert(msgsz == rc);
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

int nba::knapp::pollring_init(
        struct poll_ring *r, int32_t n, scif_epd_t epd, int node)
{
    int rc;
    assert(n > 0);
    r->len = n;
    r->alloc_bytes = ALIGN_CEIL(n * sizeof(uint64_t), PAGE_SIZE);
    r->ring =
        (uint64_t volatile *) rte_malloc_socket("poll_ring",
                                                r->alloc_bytes,
                                                PAGE_SIZE, node);
    if (r->ring == nullptr) {
        return -1;
    }
    char ringname[32];
    uintptr_t local_ring[n];
    snprintf(ringname, 32, "poll-id-pool-%d", _global_pollring_counter++);
    r->id_pool = rte_ring_create(ringname, n + 1, node, 0);
    assert(r->id_pool != nullptr);
    for (int i = 0; i < n; i++) {
        local_ring[i] = i;
    }
    rc = rte_ring_enqueue_bulk(r->id_pool, (void **)local_ring, n);
    assert(0 == rc);
    memset((void *) r->ring, 0, r->alloc_bytes);
    r->ring_ra = scif_register(epd, (void *) r->ring,
                               r->alloc_bytes, 0, SCIF_PROT_WRITE, 0);
    if (r->ring_ra < 0) {
        return -1;
    }
    return 0;
}

// vim: ts=8 sts=4 sw=4 et
