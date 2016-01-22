#ifndef DEBUG
#define DEBUG
#endif

#include <nba/core/assert.hh>
#include <nba/core/shiftedint.hh>
#include <cstdio>
#include <gtest/gtest.h>

using namespace nba;

typedef ShiftedInt<uint16_t, 4> TestInt;

TEST(CoreShiftedIntTest, Precision_Create) {
	EXPECT_THROW(TestInt(1), PrecisionLossException);

	EXPECT_NO_THROW(TestInt(16));
}

TEST(CoreShiftedIntTest, Precision_Operation) {
	auto x = TestInt(16);

	EXPECT_THROW(x = 3, PrecisionLossException);
	EXPECT_THROW(x == 3, PrecisionLossException);
	EXPECT_THROW(x != 3, PrecisionLossException);
	EXPECT_THROW(x += 3, PrecisionLossException);
	EXPECT_THROW(x + 3, PrecisionLossException);
	EXPECT_NO_THROW(x *= 3);
	EXPECT_NO_THROW(x * 3);

	EXPECT_NO_THROW(x = 32);
	EXPECT_NO_THROW(x == 32);
	EXPECT_NO_THROW(x != 32);
	EXPECT_NO_THROW(x += 32);
	EXPECT_NO_THROW(x + 32);
	EXPECT_NO_THROW(x *= 32);
	EXPECT_NO_THROW(x * 32);
}

TEST(CoreShiftedIntTest, Assignment) {
    auto x = ShiftedInt<uint16_t, 0>(123);
    x.as_value<uint32_t>();
    //EXPECT_EQ((uint32_t) x, 120) << "Precision loss due to shift should occur.";
    //auto y = ShiftedInt<uint16_t, 2>(120);
    //EXPECT_EQ((uint32_t) y, 120) << "Should be exactly same.";
}

// vim: ts=8 sts=4 sw=4 et
