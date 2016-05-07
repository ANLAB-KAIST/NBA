#ifndef __NBA_KNAPP_HOSTTYPES_HH__
#define __NBA_KNAPP_HOSTTYPES_HH__

#ifdef __MIC__
#error "This header should be used by host-side codes only."
#endif
#ifdef __NBA_KNAPP_MICTYPES_HH__
#error "Mixed use of MIC/host headers!"
#endif

#include <nba/core/intrinsic.hh>
#include <nba/core/queue.hh>
#include <nba/engines/knapp/defs.hh>
#include <nba/engines/knapp/sharedtypes.hh>
#include <cstdint>
#include <scif.h>
#include <rte_config.h>
#include <rte_memory.h>
#include <rte_mempool.h>

namespace nba { namespace knapp {

/* Forwrad decls. */
struct offload_task;
class comp_thread_ctx;

class PollRing;
class RMABuffer;

/* Host-side vDevice context */
struct vdevice {
    int device_id;

    uint32_t pipeline_depth;
    size_t ht_per_core;
    struct offload_task *tasks_in_flight;

    scif_epd_t master_epd;
    scif_epd_t data_epd;
    scif_epd_t ctrl_epd;
    struct scif_portID master_port;
    struct scif_portID mic_data_port;

    PollRing *poll_rings[KNAPP_VDEV_MAX_POLLRINGS];
    RMABuffer *rma_buffers[KNAPP_VDEV_MAX_RMABUFFERS];
    int next_poll;
};

#define SERIALIZED_LEN_INVALID 0x7fFfFfFfu
struct offload_task {
/*
    void set_offload_params(knapp_proto_t proto, int _poll_id, struct bufarray *_input_ba, struct bufarray *_result_ba, struct vdevice *_vdev) {
        apptype = proto;
        vdev = _vdev;
        poll_id = _poll_id;
        input_ba = _input_ba;
        result_ba = _result_ba;
        b_is_serialized = false;
        task_finished = false;
        serialized_len = SERIALIZED_LEN_INVALID;
        serialized = bufarray_get_va(input_ba, poll_id);
        resultbuf = bufarray_get_va(result_ba, poll_id);
    }
    inline void init(struct rte_mempool *_packet_ptrs_mempool, struct rte_mempool *_new_packet_mempool, uint32_t offload_batch_size, struct io_context *_src_ioctx) {
        count = 0;
        alloc_len = offload_batch_size;
        packet_ptrs_mempool = _packet_ptrs_mempool;
        new_packet_mempool = _new_packet_mempool;
        assert ( 0 == rte_mempool_get(packet_ptrs_mempool, (void **) &pkts) );
        assert ( 0 == rte_mempool_get_bulk(new_packet_mempool, (void **) pkts, offload_batch_size) );
        src_ioctx = _src_ioctx;
    }
    inline struct io_context *get_src_ioctx() {
        return src_ioctx;
    }
    inline int get_poll_id() {
        return poll_id;
    }
    void *get_serialized() {
        if ( b_is_serialized ) {
            return serialized;
        }
        serialize();
        return (void *)serialized;
    }
    inline size_t get_serialized_len() {
        if ( !b_is_serialized ) {
            serialize();
        }
        return serialized_len;
    }
    inline void mark_offload_start() {
        ts_offload_begin = knapp_get_usec();
    }
    inline void mark_offload_finish() {
        ts_offload_end = knapp_get_usec();
        task_finished = true;
    }
    inline bool is_finished() {
        return task_finished;
    }
    inline bool is_serialized() {
        return b_is_serialized;
    }
    inline void update_offload_ts() {
        struct offload_task_tailroom *p = get_tailroom();
        ts_proc_begin = p->ts_proc_begin;
        ts_proc_end = p->ts_proc_end;
    }
    inline struct offload_task_tailroom *get_tailroom() {
        return (struct offload_task_tailroom *)(serialized + get_result_size(apptype, count));
    }
    inline void get_timing_breakdown(uint64_t *array_us) {
        array_us[0] = ts_offload_begin;
        array_us[1] = ts_proc_begin;
        array_us[2] = ts_proc_end;
        array_us[3] = ts_offload_end;
    }
    inline uint32_t get_count() {
        return count;
    }
    inline struct packet *get_packet(int index) {
        return pkts[index];
    }
    inline void free_buffers() {
        //assert ( 0 == mp->put(serialized) );
        rte_mempool_put(packet_ptrs_mempool, (void **) pkts);
        rte_mempool_put_bulk(new_packet_mempool, (void **) pkts, alloc_len);
        pollring_put(&vdev->poll_ring, poll_id);
        serialized = NULL;
    }
    inline struct packet * get_tail() {
        return pkts[count];
    }
    inline void incr_tail() {
        count++;
    }
    inline struct vdevice *get_vdevice() {
        return vdev;
    }

    pktprocess_result_t offload_postproc(int index);

    size_t serialize();
*/
    uint64_t ts_offload_begin, ts_proc_begin, ts_proc_end, ts_offload_end;
    struct packet **pkts __rte_cache_aligned;
    struct rte_mempool *packet_ptrs_mempool;
    struct rte_mempool *new_packet_mempool;
    //knapp_proto_t apptype;
    uint32_t count;
    uint32_t alloc_len;
    int poll_id;
    struct bufarray *input_ba;
    struct bufarray *result_ba;
    bool b_is_serialized;
    bool task_finished;
    size_t serialized_len;
    off_t serialized_ra;
    uint8_t *serialized;
    uint8_t *resultbuf;
    struct comp_thread_ctx *src_ioctx;
    struct vdevice *vdev;
    //uint64_t ts_first_queued;
} __cache_aligned;

}} // endns(nba::knapp)

#endif //__NBA_KNAPP_HOSTTYPES_HH__

// vim: ts=8 sts=4 sw=4 et
