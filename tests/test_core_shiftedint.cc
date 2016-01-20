#include <nba/core/assert.hh>
#include <nba/core/shiftedint.hh>
#include <cstdio>

using namespace nba;

int main() {
    printf("TEST: core.shiftedint\n");
    {
        test_assert(sizeof(ShiftedInt<uint16_t, 0>) == sizeof(uint16_t),
                    "The size of ShfitedInt must be same to the given template.");
    }
    {
        auto x = ShiftedInt<uint16_t, 2>(123);
        test_assert((uint32_t) x == 120, "Expected precision loss due to shift.");
    }
    {
        auto x = ShiftedInt<uint16_t, 2>(120);
        test_assert((uint32_t) x == 120, "Expected exactly same value.");
    }
    return 0;
}

// vim: ts=8 sts=4 sw=4 et
