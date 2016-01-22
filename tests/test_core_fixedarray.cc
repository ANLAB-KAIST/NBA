#include <nba/core/queue.hh>
#include <nba/core/enumerate.hh>
#include <vector>
#include <string>
#include <gtest/gtest.h>

using namespace std;
using namespace nba;

TEST(CoreFixedArrayTest, Initialization) {
    FixedArray<int, -1, 3> A;
    A.push_back(1);
    EXPECT_EQ(1, A[0]);
    EXPECT_EQ(-1, A[1]);
    EXPECT_EQ(-1, A[2]);
    A.push_back(2);
    EXPECT_EQ(1, A[0]);
    EXPECT_EQ(2, A[1]);
    EXPECT_EQ(-1, A[2]);
    A.push_back(3);
    EXPECT_EQ(1, A[0]);
    EXPECT_EQ(2, A[1]);
    EXPECT_EQ(3, A[2]);
}

TEST(CoreFixedRingTest, PushBack) {
    int buf[3];
    FixedRing<int, -1> A;
    A.init(3, 0, buf);
    EXPECT_EQ(0, A.size());
    EXPECT_TRUE(A.empty());
    A.push_back(1);
    EXPECT_EQ(1, A.front());
    EXPECT_EQ(1, A[0]);
    EXPECT_EQ(1, A.size());
    EXPECT_FALSE(A.empty());
    A.push_back(2);
    EXPECT_EQ(1, A.front());
    EXPECT_EQ(1, A[0]);
    EXPECT_EQ(2, A[1]);
    EXPECT_EQ(2, A.size());
    A.push_back(3);
    EXPECT_EQ(1, A.front());
    EXPECT_EQ(1, A[0]);
    EXPECT_EQ(2, A[1]);
    EXPECT_EQ(3, A[2]);
    EXPECT_EQ(3, A.size());
}

TEST(CoreFixedRingTest, PushFront) {
    int buf[3];
    FixedRing<int, -1> B;
    B.init(3, 0, buf);
    EXPECT_EQ(0, B.size());
    EXPECT_TRUE(B.empty());
    B.push_front(1);
    EXPECT_EQ(1, B.front());
    EXPECT_EQ(1, B[0]);
    EXPECT_EQ(1, B.size());
    EXPECT_FALSE(B.empty());
    B.push_front(2);
    EXPECT_EQ(2, B.front());
    EXPECT_EQ(2, B[0]);
    EXPECT_EQ(1, B[1]);
    EXPECT_EQ(2, B.size());
    B.push_front(3);
    EXPECT_EQ(3, B.front());
    EXPECT_EQ(3, B[0]);
    EXPECT_EQ(2, B[1]);
    EXPECT_EQ(1, B[2]);
    EXPECT_EQ(3, B.size());
    int correctB[] = {3, 2, 1};
    for (auto&& p : enumerate(B))
        EXPECT_EQ(correctB[p.first], p.second);
}

TEST(CoreFixedRingTest, MixedPushBackFront) {
    int buf[3];
    FixedRing<int, -1> C;
    C.init(3, 0, buf);
    EXPECT_EQ(0, C.size());
    EXPECT_TRUE(C.empty());
    C.push_back(1);
    C.push_back(2);
    C.push_front(3);
    EXPECT_EQ(3, C[0]);
    EXPECT_EQ(1, C[1]);
    EXPECT_EQ(2, C[2]);
    EXPECT_EQ(3, C.size());
    EXPECT_FALSE(C.empty());
    int correctC[] = {3, 1, 2};
    for (auto&& p : enumerate(C))
        EXPECT_EQ(correctC[p.first], p.second);
}

TEST(CoreFixedRingTest, MixedPushPop) {
    int buf[3];
    FixedRing<int, -1> D;
    D.init(3, 0, buf);
    EXPECT_EQ(0, D.size());
    EXPECT_TRUE(D.empty());
    D.push_back(1);
    D.pop_front();
    EXPECT_EQ(0, D.size());
    EXPECT_TRUE(D.empty());
    D.push_front(1);
    D.pop_front();
    EXPECT_EQ(0, D.size());
    EXPECT_TRUE(D.empty());
    for (int x = 0; x < 20; x++) {
        // Unbalance push_back() and push_front()
        // so that internal pop_idx/push_idx goes
        // beyond the boundaries.
        D.push_back(2);
        EXPECT_EQ(2, D.front());
        D.push_front(3);
        EXPECT_EQ(3, D.front());
        D.push_back(4);
        EXPECT_EQ(3, D.front());
        int correctD[] = {3, 2, 4};
        for (auto&& p : enumerate(D))
            EXPECT_EQ(correctD[p.first], p.second);
        D.pop_front();
        EXPECT_EQ(2, D.size());
        EXPECT_FALSE(D.empty());
        EXPECT_EQ(2, D.front());
        D.pop_front();
        EXPECT_EQ(1, D.size());
        EXPECT_FALSE(D.empty());
        EXPECT_EQ(4, D.front());
        D.pop_front();
        EXPECT_EQ(0, D.size());
        EXPECT_TRUE(D.empty());
    }
}

// vim: ts=8 sts=4 sw=4 et
