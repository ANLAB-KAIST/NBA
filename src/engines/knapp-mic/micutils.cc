#include <nba/engines/knapp/defs.hh>
#include <nba/engines/knapp/micintrinsic.hh>
#include <nba/engines/knapp/micutils.hh>
#include <nba/engines/knapp/sharedtypes.hh>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <map>
#include <string>
#include <scif.h>

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

void nba::knapp::recv_ctrlmsg(
        scif_epd_t epd, uint8_t *buf, ctrl_msg_t msg,
        void *p1, void *p2, void *p3, void *p4)
{
    int size;
    int32_t msg_recvd;
    size = scif_recv(epd, buf, KNAPP_OFFLOAD_CTRLBUF_SIZE, SCIF_RECV_BLOCK);
    assert ( size == KNAPP_OFFLOAD_CTRLBUF_SIZE );
    msg_recvd = (ctrl_msg_t) *((int32_t *) buf);
    if ( msg_recvd != msg ) {
        fprintf(stderr, "Error - received ctrlmsg type (%d) does not match expected (%d)\n", msg_recvd, msg);
    }
    buf += sizeof(int32_t);
    if ( msg == OP_SET_WORKLOAD_TYPE ) {
        *((int32_t *) p1) = *((int32_t *) buf);
        buf += sizeof(int32_t);
        *((uint32_t *) p2) = *((uint32_t *) buf);
    } else if ( msg == OP_MALLOC ) {
        *((uint64_t *) p1) = *((uint64_t *) buf);
        buf += sizeof(uint64_t);
        *((uint64_t *) p2) = *((uint64_t *) buf);
        buf += sizeof(uint64_t);
        *((uint32_t *) p3) = *((uint32_t *) buf);
        buf += sizeof(uint32_t);
        *((off_t *) p4) = *((off_t *) buf);
        // Expects uint64_t remote offset and pipeline depth in return
    } else if ( msg == OP_REG_DATA ) {
        *((off_t *) buf) = *((off_t *) p1);
        // Expects ctrl_resp_t (rc) in return
    } else if ( msg == OP_REG_POLLRING ) {
        *((int32_t *) p1) = *((int32_t *) buf); // number of max poll ring elements
        buf += sizeof(int32_t);
        *((off_t *) p2) = *((off_t *) buf); // poll ring base offset
        // Expects ctrl_resp_t and remote poll-ring offset (off_t) in return
    } else if ( msg == OP_SEND_DATA ) {
        *((uint64_t *) p1) = *((uint64_t *) buf); // data size
        buf += sizeof(uint64_t);
        *((off_t *) p2) = *((off_t *) buf); //
        buf += sizeof(off_t);
        *((int32_t *) p3) = *((int32_t *) buf); // poll-ring index to use
        buf += sizeof(int32_t);
        *((int32_t *) p4) = *((int32_t *) buf);
    } else {
        log_error("Invalid control message: %d!\n", msg);
        exit(EXIT_FAILURE);
        return;
    }
}

void nba::knapp::send_ctrlresp(
        scif_epd_t epd, uint8_t *buf, ctrl_msg_t msg_recvd,
        void *p1, void *p2, void *p3, void *p4)
{
    uint8_t *buf_orig = buf;
    ctrl_msg_t msg = msg_recvd;
    int size;
    *((int32_t *) buf) = RESP_SUCCESS;
    buf += sizeof(int32_t);
    if ( msg == OP_SET_WORKLOAD_TYPE ) {

    } else if ( msg == OP_MALLOC ) {
        *((off_t *) buf) = *((off_t *) p1);
        // Expects uint64_t remote offset in return
        //buf += sizeof(off_t);
        //*((off_t *) buf) = *((off_t *) p2);
    } else if ( msg == OP_REG_DATA ) {
        // not used yet.
        //*((off_t *) buf)  = *((off_t *) p1);
        // Expects ctrl_resp_t (rc) in return
    } else if ( msg == OP_REG_POLLRING ) {
        *((off_t *) buf) = *((off_t *) p1); // mic's registered poll ring base offset
    } else if ( msg == OP_SEND_DATA ) {

    } else {
        log_error("Invalid control message: %d!\n", msg);
        exit(EXIT_FAILURE);
        return;
    }
    size = scif_send(epd, buf_orig, KNAPP_OFFLOAD_CTRLBUF_SIZE, SCIF_SEND_BLOCK);
    if ( size != KNAPP_OFFLOAD_CTRLBUF_SIZE ) {
        log_error("Error while sending ctrl msg %d - error code %d\n", msg, size);
        return;
    }
    //log_info("Sent %d bytes as ctrl resp (SUCCESS)\n", KNAPP_OFFLOAD_CTRLBUF_SIZE);
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
