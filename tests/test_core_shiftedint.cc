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
	EXPECT_THROW(TestInt(1164032), PrecisionLossException);
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

	EXPECT_THROW(x = 103, PrecisionLossException);
	EXPECT_THROW(x == 103, PrecisionLossException);
	EXPECT_THROW(x != 103, PrecisionLossException);
	EXPECT_THROW(x += 103, PrecisionLossException);
	EXPECT_THROW(x + 103, PrecisionLossException);
	EXPECT_NO_THROW(x *= 103);
	EXPECT_NO_THROW(x * 103);

	EXPECT_NO_THROW(x = 32);
	EXPECT_NO_THROW(x == 32);
	EXPECT_NO_THROW(x != 32);
	EXPECT_NO_THROW(x += 32);
	EXPECT_NO_THROW(x + 32);
	EXPECT_NO_THROW(x *= 32);
	EXPECT_NO_THROW(x * 32);
}

TEST(CoreShiftedIntTest, Assignment) {
	auto a = TestInt(0);
	auto x = TestInt(128);
	auto y = TestInt(512);
	auto z = TestInt(262144);

	EXPECT_EQ(a.as_value<uint32_t>(), 0);
	EXPECT_EQ(a.as_value<uint64_t>(), 0);

	a = 128;

	EXPECT_EQ(a.as_value<uint32_t>(), 128);
	EXPECT_EQ(a.as_value<uint64_t>(), 128);

	a = 512;

	EXPECT_EQ(a.as_value<uint32_t>(), 512);
	EXPECT_EQ(a.as_value<uint64_t>(), 512);

	a = 262144;

	EXPECT_EQ(a.as_value<uint32_t>(), 262144);
	EXPECT_EQ(a.as_value<uint64_t>(), 262144);

	a = 0;

	EXPECT_EQ(a.as_value<uint32_t>(), 0);
	EXPECT_EQ(a.as_value<uint64_t>(), 0);

	a = x;

	EXPECT_EQ(a.as_value<uint32_t>(), 128);
	EXPECT_EQ(a.as_value<uint64_t>(), 128);

	a = y;

	EXPECT_EQ(a.as_value<uint32_t>(), 512);
	EXPECT_EQ(a.as_value<uint64_t>(), 512);

	a = z;

	EXPECT_EQ(a.as_value<uint32_t>(), 262144);
	EXPECT_EQ(a.as_value<uint64_t>(), 262144);
}

TEST(CoreShiftedIntTest, Comparison) {
	auto x = TestInt(128);
	auto y = TestInt(262176);
	auto z = TestInt(262144);

	auto x2 = TestInt(128);
	auto y2 = TestInt(262176);
	auto z2 = TestInt(262144);

	EXPECT_TRUE(x == x2);
	EXPECT_TRUE(y == y2);
	EXPECT_TRUE(z == z2);

	EXPECT_FALSE(x != x2);
	EXPECT_FALSE(y != y2);
	EXPECT_FALSE(z != z2);

	EXPECT_FALSE(x == z2);
	EXPECT_FALSE(y == x2);
	EXPECT_FALSE(z == y2);

	EXPECT_TRUE(x != y2);
	EXPECT_TRUE(y != z2);
	EXPECT_TRUE(z != x2);

	EXPECT_TRUE(x == 128);
	EXPECT_TRUE(y == 262176);
	EXPECT_TRUE(z == 262144);

	EXPECT_FALSE(x != 128);
	EXPECT_FALSE(y != 262176);
	EXPECT_FALSE(z != 262144);

	EXPECT_FALSE(x == 262144);
	EXPECT_FALSE(y == 128);
	EXPECT_FALSE(z == 262176);

	EXPECT_TRUE(x != 262176);
	EXPECT_TRUE(y != 262144);
	EXPECT_TRUE(z != 128);
}

TEST(CoreShiftedIntTest, Incremental) {
	auto x = TestInt(128);
	auto y = TestInt(262176);
	auto z = TestInt(262304);

	EXPECT_EQ(x+y,z);
	EXPECT_EQ(x+=y,z);
	EXPECT_EQ(x,z);

	x = 128;
	EXPECT_EQ(x+262176,262304);
	EXPECT_EQ(x+=262176,262304);
	EXPECT_EQ(x,262304);
}

TEST(CoreShiftedIntTest, Multiplication) {
	auto x = TestInt(784);
	auto y = TestInt(912);
	auto z = TestInt(2352);

	EXPECT_EQ(x*3,z);
	EXPECT_EQ(x*=3,z);
	EXPECT_EQ(x,z);

	x = 784;
	EXPECT_EQ(x*y, 715008);
	EXPECT_EQ((x*=y), 715008);
	EXPECT_EQ(x, 715008);

	x = 784;
	EXPECT_EQ(x*912, 715008);
	EXPECT_EQ((x*=912), 715008);
	EXPECT_EQ(x, 715008);
}

TEST(CoreShiftedIntTest, Compound) {

	uint32_t A = 912;
	uint32_t B = 784;
	uint32_t C = 592;
	uint32_t D= A*B;
	uint32_t E= A*B+C;
	uint32_t F= (A+B)*C;
	auto a = TestInt(A);
	auto b = TestInt(B);
	auto c = TestInt(C);
	auto d = TestInt(D);
	auto e = TestInt(E);
	auto f = TestInt(F);



	EXPECT_EQ(a * b, d);
	EXPECT_EQ(a * b + c, e);
	EXPECT_EQ(c + a * b, e);
	EXPECT_EQ((a+b)*c, f);
	EXPECT_EQ(c*(a+b), f);

	EXPECT_EQ(a * B, D);
	EXPECT_EQ(a * B + c, E);
	EXPECT_EQ(a * b + C, E);
	EXPECT_EQ(a * B + C, E);
	EXPECT_EQ(c + a * B, E);
	EXPECT_EQ((a+B)*c, F);
	EXPECT_EQ((a+b)*C, F);
	EXPECT_EQ((a+B)*C, F);
	EXPECT_EQ(c*(a+B), F);

	EXPECT_EQ(a.as_value<uint32_t>(), A);
	EXPECT_EQ(b.as_value<uint32_t>(), B);
	EXPECT_EQ(c.as_value<uint32_t>(), C);
	EXPECT_EQ(d.as_value<uint32_t>(), D);
	EXPECT_EQ(e.as_value<uint32_t>(), E);
	EXPECT_EQ(f.as_value<uint32_t>(), F);

}

// vim: ts=8 sts=4 sw=4 et
