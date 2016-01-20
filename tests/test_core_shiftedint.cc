#include <nba/core/assert.hh>
#include <nba/core/shiftedint.hh>
#include <cstdio>
#include <gtest/gtest.h>

using namespace nba;

TEST(CoreShiftedIntTest, StorageSize) {
    EXPECT_EQ(sizeof(ShiftedInt<uint16_t, 0>), sizeof(uint16_t));
}

TEST(CoreShiftedIntTest, Assignment) {
    auto x = ShiftedInt<uint16_t, 2>(123);
    EXPECT_EQ((uint32_t) x, 120) << "Precision loss due to shift should occur.";
    auto y = ShiftedInt<uint16_t, 2>(120);
    EXPECT_EQ((uint32_t) y, 120) << "Should be exactly same.";
}

// vim: ts=8 sts=4 sw=4 et
