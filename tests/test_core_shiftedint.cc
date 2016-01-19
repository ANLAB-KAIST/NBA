#include <nba/core/assert.hh>
#include <nba/core/shiftedint.hh>
#include <cstdio>

using namespace nba;

int main() {
    printf("TEST: core.shiftedint\n");
    {
        auto x = ShiftedInt<uint16_t, 2>(123);
        test_assert((uint32_t) x == 120, "Assign-shift failed.");
    }
    {
        auto x = ShiftedInt<uint16_t, 2>(120);
        test_assert((uint32_t) x == 120, "Assign-shift failed.");
    }
}

// vim: ts=8 sts=4 sw=4 et
