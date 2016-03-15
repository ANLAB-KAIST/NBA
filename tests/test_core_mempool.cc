#include <nba/core/mempool.hh>
#include <gtest/gtest.h>

namespace nba {

class DummyMemoryPool : public MemoryPool<uintptr_t>
{
public:
    DummyMemoryPool() : MemoryPool() { }
    DummyMemoryPool(size_t max_size) : MemoryPool(max_size) { }
    DummyMemoryPool(size_t max_size, size_t align) : MemoryPool(max_size, align) { }

    bool init() { return true; /* no-op */ }

    uintptr_t get_base_ptr() const { return 0; }

    int alloc(size_t size, uintptr_t& ptr)
    {
        size_t offset;
        int ret = _alloc(size, &offset);
        if (ret == 0)
            ptr = (uintptr_t) offset;
        return ret;
    }

    void destroy() { /* no-op */ }
};

} // endns(nba)

using namespace nba;

TEST(CoreMempoolTest, Alloc) {
    DummyMemoryPool mp(100lu, 32lu);
    uintptr_t p;
    EXPECT_EQ(0, mp.alloc(50, p));
    EXPECT_EQ(0, p);
    EXPECT_EQ(64, mp.get_alloc_size());
    EXPECT_EQ(0, mp.alloc(20, p));
    EXPECT_EQ(64, p);
    EXPECT_EQ(96, mp.get_alloc_size());
    EXPECT_EQ(-ENOMEM, mp.alloc(4, p))
              << "Even if cur_pos + size <= max_size, "
                 "it should fail when aligned size exceeds max_size.";
    EXPECT_EQ(64, p) << "Pointer should not change when full.";
    EXPECT_EQ(96, mp.get_alloc_size()) << "Alloc size should not change when full.";
    mp.reset();
    EXPECT_EQ(0, mp.get_alloc_size());
}

// vim: ts=8 sts=4 sw=4 et
