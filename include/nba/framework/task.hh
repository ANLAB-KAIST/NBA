#ifndef __NBA_TASK_HH__
#define __NBA_TASK_HH__

#include <nba/core/intrinsic.hh>

namespace nba {

enum TaskTypes : unsigned {
    TASK_SINGLE_BATCH = 0,
    TASK_OFFLOAD      = 1
};

class Element;
class PacketBatch;
class OffloadTask;

struct task_tracker {
    Element* element;
    int input_port;
    bool has_results;
};

namespace Task {
    /* We use a 64-bit integer to represent a task item.
     * The first 16 bits indicates the task type and
     * the last 48 bits indicates the pointer address
     * to raw data types. */
    static const uintptr_t TYPE_MASK = 0xffff000000000000u;
    static const uintptr_t PTR_MASK  = 0x0000ffffffffffffu;
    static const uintptr_t PTR_SIZE  = 48u;

    static inline void *to_task(PacketBatch *batch)
    {
        uintptr_t p = ((uintptr_t) TASK_SINGLE_BATCH << PTR_SIZE)
                    | ((uintptr_t) batch & PTR_MASK);
        return (void *) p;
    }

    static inline void *to_task(OffloadTask *otask)
    {
        uintptr_t p = ((uintptr_t) TASK_OFFLOAD << PTR_SIZE)
                    | ((uintptr_t) otask & PTR_MASK);
        return (void *) p;
    }

    static inline unsigned get_task_type(void *qitem)
    {
        return (unsigned) ((((uintptr_t) qitem) & TYPE_MASK) >> PTR_SIZE);
    }

    static inline PacketBatch *to_packet_batch(void *qitem)
    {
        uintptr_t p = ((uintptr_t) qitem) & PTR_MASK;
        return (PacketBatch *) p;
    }

    static inline OffloadTask *to_offload_task(void *qitem)
    {
        uintptr_t p = ((uintptr_t) qitem) & PTR_MASK;
        return (OffloadTask *) p;
    }

} /* endns(nba::Task) */

} /* endns(nba) */

#endif /* __NBA_TASK_HH__ */

// vim: ts=8 sts=4 sw=4 et foldmethod=marker
