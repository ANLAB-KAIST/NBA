#ifndef __NBA_KNAPP_SHAREDTYPES_HH__
#define __NBA_KNAPP_SHAREDTYPES_HH__

#include <cstdint>
#include <scif.h>
#ifdef __MIC__
#include <nba/engines/knapp/micintrinsic.hh>
#else
#include <nba/core/intrinsic.hh>
#endif

namespace nba { namespace knapp {

struct taskitem {
    int32_t task_id;      // doubles as poll/buffer index
    uint64_t input_size;
    uint32_t num_items;
} __cache_aligned;

}} // endns(nba::knapp)

#endif // __NBA_KNAPP_SHAREDTYPES_HH__

// vim: ts=8 sts=4 sw=4 et
