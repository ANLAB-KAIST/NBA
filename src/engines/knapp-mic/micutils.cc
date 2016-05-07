#include <nba/engines/knapp/defs.hh>
#include <nba/engines/knapp/micintrinsic.hh>
#include <nba/engines/knapp/micutils.hh>
#include <nba/engines/knapp/sharedtypes.hh>
#include <nba/engines/knapp/ctrl.pb.h>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <map>
#include <string>
#include <scif.h>
#include <signal.h>
#include <poll.h>

namespace nba { namespace knapp {

}} // endns(nba::knapp);

using namespace nba::knapp;

void nba::knapp::set_cpu_mask(
        pthread_attr_t *attr, unsigned pinned_core, unsigned num_cores)
{
    size_t cpuset_sz = CPU_ALLOC_SIZE(num_cores);
    cpu_set_t *cpuset = CPU_ALLOC(num_cores);
    CPU_ZERO_S(cpuset_sz, cpuset);
    CPU_SET_S(pinned_core, cpuset_sz, cpuset);
    pthread_attr_setaffinity_np(attr, cpuset_sz, cpuset);
    CPU_FREE(cpuset);
}

bool nba::knapp::recv_ctrlmsg(
        scif_epd_t epd, CtrlRequest &req, sigset_t *orig_sigmask)
{
    int rc;
    char buf[1024];
    uint32_t msgsz = 0;
    req.Clear();
    struct pollfd p = {epd, POLLIN, 0};
    rc = ppoll(&p, 1, nullptr, orig_sigmask);
    if (rc == -1) // errno is set
        return false;
    rc = scif_recv(epd, &msgsz, sizeof(msgsz), SCIF_RECV_BLOCK);
    if (rc == -1) // errno is set
        return false;
    assert(sizeof(msgsz) == rc);
    assert(msgsz < 1024);
    rc = scif_recv(epd, buf, msgsz, SCIF_RECV_BLOCK);
    if (rc == -1) // errno is set
        return false;
    assert(msgsz == rc);
    req.ParseFromArray(buf, msgsz);
    return true;
}

bool nba::knapp::send_ctrlresp(scif_epd_t epd, const CtrlResponse &resp)
{
    int rc;
    char buf[1024];
    uint32_t msgsz = resp.ByteSize();
    assert(msgsz < 1024);
    resp.SerializeToArray(buf, msgsz);
    rc = scif_send(epd, &msgsz, sizeof(msgsz), SCIF_SEND_BLOCK);
    if (rc == -1) // errno is set
        return false;
    assert(sizeof(msgsz) == rc);
    rc = scif_send(epd, buf, msgsz, SCIF_SEND_BLOCK);
    if (rc == -1) // errno is set
        return false;
    assert(msgsz == rc);
    return true;
}

// vim: ts=8 sts=4 sw=4 et
