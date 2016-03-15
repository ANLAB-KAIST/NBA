#include <nba/core/enumerate.hh>
#include <vector>
#include <string>
#include <gtest/gtest.h>

// Test cases are adopted from
// https://github.com/minjang/minjang.github.io/blob/master/assets/2016/python_like_enumerate/python_like.cpp

using namespace std;
using namespace nba;

TEST(CoreEnumerateTest, ForwardIterator) {
    vector<string> A = {"foo", "bar", "baz"};
    const vector<string> Adup = {"foofoo", "barbar", "bazbaz"};
    vector<string> B = {"foo", "bar", "baz"};
    const vector<string> Borig = {"foo", "bar", "baz"};
    size_t idx = 0;
    // Get elements via reference. A is modified.
    for (pair<size_t, string&> p : enumerate(A)) {
        EXPECT_EQ(idx, p.first) << "Enumerated index must match with induction-based index.";
        p.second += p.second;
        idx ++;
    }
    for (pair<size_t, const string&> p : enumerate(A)) {
        EXPECT_EQ(Adup[p.first], p.second) << "A should have been modified.";
    }
    // Get elements via value (copy). B is not modified.
    for (pair<size_t, string> p : enumerate(B)) {
        p.second += p.second;
    }
    for (auto p : enumerate(B)) { // type of p: pair<size_t, string&>
        EXPECT_EQ(Borig[p.first], p.second) << "B should not have been modified.";
    }
}

TEST(CoreEnumerateTest, Array) {
    string C[] = {"foo", "bar", "baz"};
    const string Cdup[] = {"foofoo", "barbar", "bazbaz"};
    size_t idx = 0;
    // Generally, auto&& is efficient since it does not copy pair instances.
    // In this case, decltype(p) == pair<size_t, string&>&&
    for (auto&& p : enumerate(C, 100)) {
        EXPECT_EQ(idx + 100, p.first) << "Enumerated index must match with induction-based index.";
        p.second += p.second;
        idx ++;
    }
    for (auto&& p : enumerate(C, 100)) {
        EXPECT_EQ(Cdup[p.first - 100], p.second) << "auto&& should allow modification.";
    }
}

TEST(CoreEnumerateTest, ConstIterator) {
    const string E[] = {"foo", "bar", "baz"};
    size_t idx = 0;
    // In this case, decltype(p) == pair<size_t, string const&>&&
    for (auto&& p : enumerate(E)) {
        p.first += 1;
        EXPECT_EQ(idx + 1, p.first) << "Index should be modifiable.";
        // p.second is const
        idx ++;
    }
}

TEST(CoreEnumerateTest, RValueIterator) {
    const vector<string> T = {"foo", "bar", "baz"};
    size_t idx = 0;
    // Object creation is an rvalue.
    for (auto&& p : enumerate(vector<string>{"foo", "bar", "baz"})) {
        EXPECT_EQ(idx, p.first);
        EXPECT_EQ(T[idx], p.second);
        idx ++;
    }
    // Function return value is also an rvalue.
    // decltype(p) == pair<size_t, string&>&&
    auto create = []()->vector<string> { return {"foo", "bar", "baz"}; };
    idx = 0;
    for (auto&& p : enumerate(create())) {
        EXPECT_EQ(idx, p.first);
        EXPECT_EQ(T[idx], p.second);
        idx ++;
    }
}

TEST(CoreEnumerateTest, InitializerList) {
    const vector<string> T = {"foo", "bar", "baz"};
    size_t idx = 0;
    for (auto&& p : enumerate({"foo", "bar", "baz"})) {
        EXPECT_EQ(idx, p.first);
        EXPECT_EQ(T[idx], p.second);
        idx ++;
    }
}

// vim: ts=8 sts=4 sw=4 et
