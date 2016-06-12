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

// vim: ts=8 sts=4 sw=4 et
