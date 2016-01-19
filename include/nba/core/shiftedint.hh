#ifndef __NBA_CORE_SHIFTEDINT_HH__
#define __NBA_CORE_SHIFTEDINT_HH__

#include <cstdint>
#include <type_traits>

namespace nba {

template<typename Int, unsigned Shift>
class ShiftedInt
{
    static_assert(std::is_integral<Int>::value, "Integer type required.");

    Int i;

public:
    ShiftedInt(Int initial) : i(initial) { }

    ShiftedInt& operator= (Int v) { i = (v >> Shift); return *this; }
    operator uint32_t() { return (uint32_t) (i << Shift); }
};

} /* endns(nba) */

#endif /* __NBA_CORE_SHIFTEDINT_HH__ */

// vim: ts=8 sts=4 sw=4 et
