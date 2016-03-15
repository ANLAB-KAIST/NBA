#ifndef __NBA_CORE_ACCUMIDX_HH__
#define __NBA_CORE_ACCUMIDX_HH__

#include <nba/core/errors.hh>
#include <type_traits>

#ifndef __CUDACC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

namespace nba {

template<typename T>
__host__ __device__ static inline nba::error_t get_accum_idx(
    const T *group_counts,
    const T num_groups,
    const T global_idx,
    T &group_idx,
    T &item_idx)
{
    static_assert(std::is_integral<T>::value, "Integer type required.");
    T sum = 0;
    T i;
    bool found = false;
    for (i = 0; i < num_groups; i++) {
        if (global_idx >= sum && global_idx < sum + group_counts[i]) {
            item_idx = global_idx - sum;
            found = true;
            break;
        }
        sum += group_counts[i];
    }
    if (found)
        group_idx = i;
    else
        return NBA_NOT_FOUND;
    return NBA_SUCCESS;
}

} // endns(nba)

#endif

// vim: ts=8 sts=4 sw=4 et
