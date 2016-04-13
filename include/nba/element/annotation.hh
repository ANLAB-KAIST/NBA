#ifndef __NBA_ANNOTATION_HH__
#define __NBA_ANNOTATION_HH__

#include <cstdint>
#include <nba/framework/config.hh>
#include <cassert>

namespace nba {

/* Predefined per-packet annotations */
enum PacketAnnotationKind : unsigned
{
    NBA_ANNO_IFACE_IN = 0,
    NBA_ANNO_IFACE_OUT,
    NBA_ANNO_TIMESTAMP,
    NBA_ANNO_BATCH_ID,
    NBA_ANNO_IPSEC_FLOW_ID,
    NBA_ANNO_IPSEC_IV1,
    NBA_ANNO_IPSEC_IV2,

    //End of PacketAnnotationKind
    NBA_MAX_ANNOTATION_SET_SIZE
};

enum BatchAnnotationKind : unsigned
{
    NBA_BANNO_LB_DECISION = 0,

    //End of BatchAnnotationKind
    NBA_MAX_BANNOTATION_SET_SIZE
};

struct annotation_set {
    uint64_t bitmask;
    int64_t values[NBA_MAX_ANNOTATION_SET_SIZE];
    static_assert((unsigned) NBA_MAX_ANNOTATION_SET_SIZE >= (unsigned) NBA_MAX_BANNOTATION_SET_SIZE,
                  "The number of packet annotations must be larger than that of batch annotations.");
};

#define anno_isset(anno_item, anno_id)  (anno_item != nullptr && (anno_item)->bitmask & (1 << anno_id))

static inline void anno_set(struct annotation_set *anno_item,
                            unsigned anno_id,
                            int64_t value)
{
    anno_item->bitmask |= (1 << anno_id);
    anno_item->values[anno_id] = value;
}

static inline int64_t anno_get(struct annotation_set *anno_item,
                               unsigned anno_id)
{
    return anno_item->values[anno_id];
}

static inline void anno_copy(struct annotation_set *dst,
                             struct annotation_set *src)
{
    dst->bitmask = src->bitmask;
    uint64_t mask = src->bitmask;
    while (mask != 0) {
        const unsigned anno_idx = __builtin_ctzll(mask);
        mask &= ~(1llu << anno_idx);
        dst->values[anno_idx] = src->values[anno_idx];
    }
}

// TODO: implement custom annotation mapping

}

#endif

// vim: ts=8 sts=4 sw=4 et
