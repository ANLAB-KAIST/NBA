#ifndef __NBA_KNAPP_TYPES_HH__
#define __NBA_KNAPP_TYPES_HH__

#include <nba/core/queue.hh>
#include <scif.h>

namespace nba {
namespace knapp {

/* Forwrad decls. */
struct offload_task;

struct vdevice {
    int device_id;

    int next_poll;

    scif_epd_t data_epd;
    scif_epd_t ctrl_epd;
    uint32_t pipeline_depth;
    size_t ht_per_core;
    struct offload_task *tasks_in_flight;

    struct scif_portID local_dataport;
    struct scif_portID local_ctrlport;
    struct scif_portID remote_dataport;
    struct scif_portID remote_ctrlport;

    uint8_t *ctrlbuf;

    FixedRing<int> *cores;
};

struct offload_task {
};

} // endns(knapp)
} // endns(nba)

#endif //__NBA_KNAPP_TYPES_HH__

// vim: ts=8 sts=4 sw=4 et
