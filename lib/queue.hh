#ifndef __NSHADER_QUEUE_HH__
#define __NSHADER_QUEUE_HH__

#include <vector>
#include <cassert>

#include <rte_malloc.h>



template<class T, T const default_value, size_t max_size>
class FixedArray; 

template<class T, T const default_value, size_t max_size>
class FixedArrayIterator
{
public:
    FixedArrayIterator(const FixedArray<T, default_value, max_size>* p_array, unsigned pos)
    : _pos(pos), _p_array(p_array)
    { }

    bool operator!= (const FixedArrayIterator<T, default_value, max_size> &other) const
    {
        return _pos != other._pos;
    }

    T operator*() const;

    const FixedArrayIterator<T, default_value, max_size>& operator++ ()
    {
        ++ _pos;
        return *this;
    }

private:
    unsigned _pos;
    const FixedArray<T, default_value, max_size> *_p_array;
};

template<class T, T const default_value, size_t max_size>
class FixedArray
{
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

    FixedArrayIterator<T, default_value, max_size> begin() const
    {
        return FixedArrayIterator<T, default_value, max_size>(this, 0);
    }

    FixedArrayIterator<T, default_value, max_size> end() const
    {
        return FixedArrayIterator<T, default_value, max_size>(this, size());
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

template<class T, T const default_value, size_t max_size>
T FixedArrayIterator<T, default_value, max_size>::operator* () const
{
    return _p_array->get(_pos);
}

template<class T, T const default_value>
class FixedRing;

template<class T, T const default_value>
class FixedRingIterator
{
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
    FixedRingIterator(const FixedRing<T, default_value>* p_ring, unsigned pos)
    : _pos(pos), _p_ring(p_ring)
    { }

    bool operator!= (const FixedRingIterator<T, default_value> &other) const
    {
        return _pos != other._pos;
    }

    T operator*() const;

    const FixedRingIterator<T, default_value>& operator++ ()
    {
        ++ _pos;
        return *this;
    }

private:
    unsigned _pos;
    const FixedRing<T, default_value> *_p_ring;
};

template<class T, T const default_value>
class FixedRing
{
public:
    FixedRing()
        : v_(nullptr), push_idx(0), pop_idx(0), count(0), max_size(0)
    {
        /* Default constructor. You must explicitly call init() to use the instance. */
    }

    FixedRing(size_t max_size, int numa_node = 0)
        : v_(nullptr), push_idx(0), pop_idx(0), count(0), max_size(max_size)
    {
        init(max_size, numa_node);
    }

    virtual ~FixedRing()
    {
        if (v_ != nullptr)
            rte_free(v_);
    }

    void init(size_t max_size, int numa_node = 0)
    {
        assert(max_size > 0);
        this->count = 0;
        this->max_size = max_size;
        v_ = (T*) rte_malloc_socket("fixedring", sizeof(T) * max_size, 64, numa_node);
        assert(v_ != nullptr);
    }

    void push_back(T t)
    {
        assert(count < max_size);
        v_[push_idx] = t;
        push_idx = (push_idx + 1) % max_size;
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

    FixedRingIterator<T, default_value> begin() const
    {
        return FixedRingIterator<T, default_value>(this, 0);
    }

    FixedRingIterator<T, default_value> end() const
    {
        return FixedRingIterator<T, default_value>(this, size());
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
    size_t push_idx;
    size_t pop_idx;
    size_t count;
    size_t max_size;
};

template<class T, T const default_value>
T FixedRingIterator<T, default_value>::operator* () const
{
    return _p_ring->get(_pos);
}


#endif
// vim: ts=8 sts=4 sw=4 et
