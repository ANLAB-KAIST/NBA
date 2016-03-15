#include <nba/core/errors.hh>
#include <nba/core/accumidx.hh>
#include <gtest/gtest.h>

using namespace std;
using namespace nba;

TEST(AccumIdxTest, Count) {
    const unsigned num_groups = 4;
    const unsigned groups[num_groups] = { 35, 1, 0, 21 };
    unsigned group_idx = 0, item_idx = 0;
    EXPECT_EQ(NBA_SUCCESS, get_accum_idx(groups, num_groups, 0u, group_idx, item_idx));
    EXPECT_EQ(0, group_idx);
    EXPECT_EQ(0, item_idx);
    EXPECT_EQ(NBA_SUCCESS, get_accum_idx(groups, num_groups, 1u, group_idx, item_idx));
    EXPECT_EQ(0, group_idx);
    EXPECT_EQ(1, item_idx);
    EXPECT_EQ(NBA_SUCCESS, get_accum_idx(groups, num_groups, 17u, group_idx, item_idx));
    EXPECT_EQ(0, group_idx);
    EXPECT_EQ(17, item_idx);
    EXPECT_EQ(NBA_SUCCESS, get_accum_idx(groups, num_groups, 34u, group_idx, item_idx));
    EXPECT_EQ(0, group_idx);
    EXPECT_EQ(34, item_idx);
    EXPECT_EQ(NBA_SUCCESS, get_accum_idx(groups, num_groups, 35u, group_idx, item_idx));
    EXPECT_EQ(1, group_idx);
    EXPECT_EQ(0, item_idx);
    EXPECT_EQ(NBA_SUCCESS, get_accum_idx(groups, num_groups, 36u, group_idx, item_idx));
    EXPECT_EQ(3, group_idx);
    EXPECT_EQ(0, item_idx);
    EXPECT_EQ(NBA_SUCCESS, get_accum_idx(groups, num_groups, 37u, group_idx, item_idx));
    EXPECT_EQ(3, group_idx);
    EXPECT_EQ(1, item_idx);
    EXPECT_EQ(NBA_SUCCESS, get_accum_idx(groups, num_groups, 56u, group_idx, item_idx));
    EXPECT_EQ(3, group_idx);
    EXPECT_EQ(20, item_idx);
    group_idx = 7;
    item_idx = 7;
    EXPECT_EQ(NBA_NOT_FOUND, get_accum_idx(groups, num_groups, 57u, group_idx, item_idx));
    EXPECT_EQ(7, group_idx);
    EXPECT_EQ(7, item_idx);
}

// vim: ts=8 sts=4 sw=4 et
