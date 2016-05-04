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

static std::map<nba::knapp::ctrl_msg_t, std::string>
ctrltype_to_ctrlstring = {
    { OP_SET_WORKLOAD_TYPE, "OP_SET_WORKLOAD_TYPE" },
    { OP_MALLOC, "OP_MALLOC" },
    { OP_REG_DATA, "OP_REG_DATA" },
    { OP_REG_POLLRING, "OP_REG_POLLRING" },
    { OP_SEND_DATA, "OP_SEND_DATA" }
};

}} // endns(nba::knapp);

using namespace nba::knapp;

extern int core_util[][KNAPP_MAX_THREADS_PER_CORE];

int nba::knapp::get_least_utilized_ht(int pcore)
{
    int min_util = 0x7fffffff;
    int ret = 0;
    assert ( pcore >= 0 && pcore < KNAPP_NUM_CORES );
    for ( int i = 0; i < KNAPP_MAX_THREADS_PER_CORE; i++ ) {
        if ( core_util[pcore][i] == 0 ) {
            return i;
        }
        if ( core_util[pcore][i] < min_util ) {
            min_util = core_util[pcore][i];
            ret = i;
        }
    }
    return ret;
}

bool nba::knapp::recv_ctrlmsg(scif_epd_t epd, CtrlRequest &req, sigset_t *orig_sigmask)
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

int nba::knapp::pollring_init(
    struct poll_ring *r, int32_t n, scif_epd_t epd)
{
    assert(n > 0);
    r->len = n;
    r->alloc_bytes = ALIGN_CEIL(n * sizeof(uint64_t), PAGE_SIZE);
    r->ring = (uint64_t volatile *) _mm_malloc(r->alloc_bytes, PAGE_SIZE);
    if (r->ring == nullptr) {
        return -1;
    }
    memset((void *) r->ring, 0, r->alloc_bytes);
    r->ring_ra = scif_register(epd, (void *) r->ring,
                               r->alloc_bytes, 0, SCIF_PROT_WRITE, 0);
    if (r->ring_ra < 0) {
        return -1;
    }
    return 0;
}

// vim: ts=8 sts=4 sw=4 et
