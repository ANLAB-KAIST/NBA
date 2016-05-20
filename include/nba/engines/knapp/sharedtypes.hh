#ifndef __NBA_KNAPP_SHAREDTYPES_HH__
#define __NBA_KNAPP_SHAREDTYPES_HH__

#include <cstdint>
#include <scif.h>
#include <nba/engines/knapp/defs.hh>
#ifdef __MIC__
#include <nba/engines/knapp/micintrinsic.hh>
#else
#include <nba/core/intrinsic.hh>
#endif
#include <nba/framework/datablock_shared.hh>

namespace nba { namespace knapp {

#define NBA_KNAPP_MAX_D2HCOPIES 8

struct taskitem {
    uint32_t task_id;
    uint32_t num_items;
    uint32_t num_args;
    uint64_t kernel_id;
    void *args[KNAPP_MAX_KERNEL_ARGS];
} __cache_aligned;

struct d2hcopy {
    uint16_t num_copies;
    uint32_t buffer_id[NBA_KNAPP_MAX_D2HCOPIES];
    uint32_t offset[NBA_KNAPP_MAX_D2HCOPIES];
    uint32_t size[NBA_KNAPP_MAX_D2HCOPIES];
} __cache_aligned;

}} // endns(nba::knapp)

#endif // __NBA_KNAPP_SHAREDTYPES_HH__

// vim: ts=8 sts=4 sw=4 et
