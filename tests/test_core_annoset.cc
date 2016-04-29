#include <nba/element/annotation.hh>
#include <gtest/gtest.h>

using namespace nba;

TEST(CoreAnnoSetTest, GetSet) {
    struct annotation_set a;
    memset(&a, 0, sizeof(struct annotation_set));
    anno_set(&a, 0, 123);
    anno_set(&a, 1, -1);
    anno_set(&a, 5, 2394234);
    anno_set(&a, 6, -19357);
    EXPECT_EQ(99,      a.bitmask);
    EXPECT_EQ(123,     a.values[0]);
    EXPECT_EQ(-1,      a.values[1]);
    EXPECT_EQ(2394234, a.values[5]);
    EXPECT_EQ(-19357,  a.values[6]);
    EXPECT_EQ(123,     anno_get(&a, 0));
    EXPECT_EQ(-1,      anno_get(&a, 1));
    EXPECT_EQ(2394234, anno_get(&a, 5));
    EXPECT_EQ(-19357,  anno_get(&a, 6));
}

TEST(CoreAnnoSetTest, Enumerate) {
    EXPECT_TRUE(1);
}

TEST(CoreAnnoSetTest, Copy) {
    struct annotation_set src;
    struct annotation_set dst;
    memset(&src, 0, sizeof(struct annotation_set));
    anno_set(&src, 0, 123);
    anno_set(&src, 1, -1);
    anno_set(&src, 5, 2394234);
    anno_set(&src, 6, -19357);
    anno_copy(&dst, &src);
    EXPECT_EQ(99,      dst.bitmask);
    EXPECT_EQ(123,     dst.values[0]);
    EXPECT_EQ(-1,      dst.values[1]);
    EXPECT_EQ(2394234, dst.values[5]);
    EXPECT_EQ(-19357,  dst.values[6]);
    EXPECT_EQ(123,     anno_get(&dst, 0));
    EXPECT_EQ(-1,      anno_get(&dst, 1));
    EXPECT_EQ(2394234, anno_get(&dst, 5));
    EXPECT_EQ(-19357,  anno_get(&dst, 6));
}

// vim: ts=8 sts=4 sw=4 et
