#ifndef __NBA_CORE_SHIFTEDINT_HH__
#define __NBA_CORE_SHIFTEDINT_HH__

#include <cstdint>
#include <type_traits>
#include <limits>
#include <cassert>
#include <exception>

#ifndef __CUDACC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

namespace nba {

#ifdef DEBUG
class PrecisionLossException : public std::exception
{
private:
    const char* msg;
public:
    PrecisionLossException(const char* msg) {
        this->msg = msg;
    }
    virtual const char* what() const throw() {
        return this->msg;
    }
};
#endif

template<typename Int, unsigned Shift>
class ShiftedInt
{
private:
    typedef uint64_t LARGE_INT;
    static_assert(std::is_integral<Int>::value, "Integer type required.");

    Int shifted_value;

    public:
    template<typename InputInt>
    ShiftedInt(const InputInt& initial) {
        static_assert(std::is_integral<InputInt>::value, "Integer type required.");
#ifdef DEBUG
        if(initial > ((LARGE_INT)std::numeric_limits<Int>::max() << Shift))
            throw PrecisionLossException("input type is too large");
        if((initial & (((LARGE_INT)1 << Shift) -1)) != 0)
            throw PrecisionLossException("lower digits are lost");
#endif
        shifted_value = (initial >> Shift);
    }

    //copy constructor
    ShiftedInt(const ShiftedInt& orig) {
        shifted_value = orig.shifted_value;
    }

    template<typename InputInt>
    ShiftedInt& operator= (const InputInt& v) {
        static_assert(std::is_integral<InputInt>::value, "Integer type required.");
#ifdef DEBUG
        if(v > ((LARGE_INT)std::numeric_limits<Int>::max() << Shift))
            throw PrecisionLossException("input type is too large");
        if((v & (((LARGE_INT)1 << Shift) -1)) != 0)
            throw PrecisionLossException("lower digits are lost");
#endif
        shifted_value = (Int)(v >> Shift);
        return *this;
    }

    template<typename InputInt>
    ShiftedInt& operator+= (const InputInt& v) {
        static_assert(std::is_integral<InputInt>::value, "Integer type required.");
#ifdef DEBUG
        if(v > ((LARGE_INT)std::numeric_limits<Int>::max() << Shift))
            throw PrecisionLossException("input type is too large");
        if((v & (((LARGE_INT)1 << Shift) -1)) != 0)
            throw PrecisionLossException("lower digits are lost");
#endif
        shifted_value += (Int)(v >> Shift);
        return *this;
    }

    template<typename InputInt>
    const ShiftedInt operator+ (const InputInt& v) const {
        static_assert(std::is_integral<InputInt>::value, "Integer type required.");
#ifdef DEBUG
        if(v > ((LARGE_INT)std::numeric_limits<Int>::max() << Shift))
            throw PrecisionLossException("input type is too large");
        if((v & (((LARGE_INT)1 << Shift) -1)) != 0)
            throw PrecisionLossException("lower digits are lost");
#endif
        ShiftedInt ret(*this);
        ret.shifted_value += (Int)(v >> Shift);
        return ret;
    }

    template<typename InputInt>
    ShiftedInt& operator*= (const InputInt& v) {
        static_assert(std::is_integral<InputInt>::value, "Integer type required.");
#ifdef DEBUG
        if(v > ((LARGE_INT)std::numeric_limits<Int>::max() << Shift))
            throw PrecisionLossException("input type is too large");
#endif
        shifted_value *= (Int)(v);
        return *this;
    }

    template<typename InputInt>
    const ShiftedInt operator* (const InputInt& v) const {
        static_assert(std::is_integral<InputInt>::value, "Integer type required.");
#ifdef DEBUG
        if(v > ((LARGE_INT)std::numeric_limits<Int>::max() << Shift))
            throw PrecisionLossException("input type is too large");
#endif
        ShiftedInt ret(*this);
        ret.shifted_value *= (Int)(v);
        return ret;
    }

    template<typename InputInt>
    bool operator== (const InputInt& v) const {
        static_assert(std::is_integral<InputInt>::value, "Integer type required.");
#ifdef DEBUG
        if(v > ((LARGE_INT)std::numeric_limits<Int>::max() << Shift))
            throw PrecisionLossException("input type is too large");
        if((v & (((LARGE_INT)1 << Shift) -1)) != 0)
            throw PrecisionLossException("lower digits are lost");
#endif
        return shifted_value == (Int)(v >> Shift);
    }

    template<typename InputInt>
    bool operator!= (const InputInt& v) const {
        static_assert(std::is_integral<InputInt>::value, "Integer type required.");
#ifdef DEBUG
        if(v > ((LARGE_INT)std::numeric_limits<Int>::max() << Shift))
            throw PrecisionLossException("input type is too large");
        if((v & (((LARGE_INT)1 << Shift) -1)) != 0)
            throw PrecisionLossException("lower digits are lost");
#endif
        return shifted_value != (Int)(v >> Shift);
    }


    ShiftedInt& operator= (const ShiftedInt& v) {
        shifted_value = v.shifted_value;
        return *this;
    }

    ShiftedInt& operator+= (const ShiftedInt& v) {
        shifted_value += v.shifted_value;
        return *this;
    }

    const ShiftedInt operator+ (const ShiftedInt& v) const {
        ShiftedInt ret(*this);
        ret.shifted_value += v.shifted_value;
        return ret;
    }

    ShiftedInt& operator*= (const ShiftedInt& v) {
        shifted_value *= v.shifted_value;
        shifted_value <<= Shift;
        return *this;
    }

    const ShiftedInt operator* (const ShiftedInt& v) const {
        ShiftedInt ret(*this);
        ret.shifted_value *= v.shifted_value;
        ret.shifted_value <<= Shift;
        return ret;
    }

    bool operator== (const ShiftedInt& v) const {
        return shifted_value == v.shifted_value;
    }

    bool operator!= (const ShiftedInt& v) const {
        return shifted_value != v.shifted_value;
    }

    template<typename ReturnInt>
    __host__ __device__ inline ReturnInt as_value() const {
        static_assert(std::numeric_limits<ReturnInt>::max()
                >= ((LARGE_INT)std::numeric_limits<Int>::max() << Shift),
                "return type is not large enough.");
        return (ReturnInt) (shifted_value << Shift);
    }
};

// Changing this to ShiftedInt<uint32_t, 0> or uint32_t exactly
// reproduces the performance drop when we first changed offset types.
typedef ShiftedInt<uint16_t, 2> dev_offset_t;

} /* endns(nba) */

#endif /* __NBA_CORE_SHIFTEDINT_HH__ */

// vim: ts=8 sts=4 sw=4 et
