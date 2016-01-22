#ifndef __NBA_QUEUE_HH__
#define __NBA_QUEUE_HH__

#include <vector>
#include <cassert>

#include <rte_malloc.h>

namespace nba {

template<typename T, T const default_value, size_t max_size>
class FixedArray;

template<typename T, T const default_value, size_t max_size>
class FixedArrayIterator
{
    using ContainerType = FixedArray<T, default_value, max_size>;

    const ContainerType* const _c;
    unsigned _pos;

public:
    FixedArrayIterator(const ContainerType* const c, unsigned pos)
    : _c(c), _pos(pos)
    { }

    bool operator!= (const FixedArrayIterator& other) const
    {
        return _pos != other._pos;
    }

    const FixedArrayIterator& operator++ ()
    {
        ++ _pos;
        return *this;
    }

    T operator* () const
    {
        return _c->get(_pos);
    }
};

template<typename T, T const default_value, size_t max_size>
class FixedArray
{
    using IterType = FixedArrayIterator<T, default_value, max_size>;

public:
    FixedArray() : count(0)
    {
        /* Default constructor. You must explicitly call init() to use the instance. */
    }

    virtual ~FixedArray() { }

    void push_back(T t)
    {
        assert(count < max_size);
        values[count] = t;
        count ++;
    }

    void clear()
    {
        count = 0;
    }

    T at(unsigned i) const
    {
        if (i >= count)
            return default_value;
        return values[i];
    }

    T get(unsigned i) const
    {
        return at(i);
    }

    T operator[](const unsigned& i) const
    {
        return at(i);
    }

    IterType begin() const
    {
        return IterType(this, 0);
    }

    IterType end() const
    {
        return IterType(this, size());
    }

    bool empty() const
    {
        return count == 0;
    }

    size_t size() const
    {
        return count;
    }

private:
    size_t count;
    T values[max_size];
};

template<class T, T const default_value>
class FixedRing;

template<class T, T const default_value>
class FixedRingIterator
{
    using ContainerType = FixedRing<T, default_value>;

    const ContainerType *_p_ring;
    unsigned _pos;

public:
    /* This class is to support ranged iteration in C++11.
     * Note that iteration is not thread-safe.
     *
     * For example:
     *
     *   FixedRing<void*> ring();
     *   ...
     *   for (void *p : ring) {
     *       printf("%p\n", p);
     *   }
     */
    FixedRingIterator(const ContainerType* const p_ring, unsigned pos)
    : _p_ring(p_ring), _pos(pos)
    { }

    bool operator!= (const FixedRingIterator& other) const
    {
        return _pos != other._pos;
    }

    const FixedRingIterator& operator++ ()
    {
        ++ _pos;
        return *this;
    }

    T operator* () const
    {
        return _p_ring->get(_pos);
    }
};

template<class T, T const default_value>
class FixedRing
{
    using IterType = FixedRingIterator<T, default_value>;

public:
    FixedRing()
        : v_(nullptr), is_external(false), push_idx(0), pop_idx(0), count(0), max_size(0)
    {
        /* Default constructor. You must explicitly call init() to use the instance. */
    }

    FixedRing(size_t max_size, int numa_node = 0, T *xmem = nullptr)
        : v_(nullptr), is_external(false), push_idx(0), pop_idx(0), count(0), max_size(max_size)
    {
        init(max_size, numa_node, xmem);
    }

    virtual ~FixedRing()
    {
        if (v_ != nullptr && !is_external)
            rte_free(v_);
    }

    void init(size_t max_size, int numa_node = 0, T *xmem = nullptr)
    {
        assert(max_size > 0);
        this->count = 0;
        this->max_size = max_size;
        if (xmem == nullptr) {
            v_ = (T*) rte_malloc_socket("fixedring", sizeof(T) * max_size, 64, numa_node);
            is_external = false;
        } else {
            v_ = xmem;
            is_external = true;
        }
        assert(v_ != nullptr);
    }

    void push_back(T t)
    {
        assert(count < max_size);
        v_[push_idx] = t;
        push_idx = (push_idx + 1) % max_size;
        count ++;
    }

    void push_front(T t)
    {
        assert(count < max_size);
        size_t new_pop_idx = (max_size + pop_idx - 1) % max_size;
        v_[new_pop_idx] = t;
        pop_idx = new_pop_idx;
        count ++;
    }

    T front() const
    {
        if (!empty())
            return v_[pop_idx];
        return default_value;
    }

    T at(unsigned i) const
    {
        if (i >= count)
            return default_value;
        return v_[(pop_idx + i) % max_size];
    }

    T get(unsigned i) const
    {
        return at(i);
    }

    T operator[](const unsigned& i) const
    {
        return at(i);
    }

    IterType begin() const
    {
        return IterType(this, 0);
    }

    IterType end() const
    {
        return IterType(this, size());
    }

    void pop_front()
    {
        if (!empty()) {
            v_[pop_idx] = default_value;
            pop_idx = (pop_idx + 1) % max_size;
            count --;
        }
    }

    bool empty() const
    {
        return (push_idx == pop_idx) && (count == 0);
    }

    size_t size() const
    {
        return count;
    }

private:
    T *v_;
    bool is_external;
    size_t push_idx;
    size_t pop_idx;
    size_t count;
    size_t max_size;
};

} /* endns(nba) */

#endif
// vim: ts=8 sts=4 sw=4 et
