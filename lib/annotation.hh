#ifndef __NSHADER_ANNOTATION_HH__
#define __NSHADER_ANNOTATION_HH__

#include <cstdint>

namespace nshader {

/* Predefined per-packet annotations */
#define NSHADER_ANNO_IFACE_IN   (0)
#define NSHADER_ANNO_IFACE_OUT  (1)
#define NSHADER_ANNO_TIMESTAMP  (2)
#define NSHADER_ANNO_BATCH_ID   (3)
#define NSHADER_ANNO_IPSEC_FLOW_ID (4)
#define NSHADER_ANNO_IPSEC_IV1  (5)
#define NSHADER_ANNO_IPSEC_IV2  (6)

/* Predefined per-batch annotations */
#define NSHADER_BANNO_LB_DECISION (0)

struct annotation_set {
    uint64_t bitmask;
    int64_t values[NSHADER_MAX_ANNOTATION_SET_SIZE];
};

#define anno_isset(anno_item, anno_id)  (anno_item != nullptr && (anno_item)->bitmask & (1 << anno_id))

static inline void anno_set(struct annotation_set *anno_item,
                            unsigned anno_id,
                            int64_t value)
{
    anno_item->bitmask |= (1 << anno_id);
    anno_item->values[anno_id] = value;
}

#define anno_get(anno_item, anno_id)    ((anno_item)->values[anno_id])

// TODO: implement custom annotation mapping

}

#endif

// vim: ts=8 sts=4 sw=4 et
