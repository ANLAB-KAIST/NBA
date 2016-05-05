#ifndef __NBA_KNAPP_MICINTRINSIC_HH__
#define __NBA_KNAPP_MICINTRINSIC_HH__

#ifndef __MIC__
#error "This header should be used by MIC-side codes only."
#endif

#include <cstdint>

#ifdef CACHE_LINE_SIZE
#undef CACHE_LINE_SIZE
#endif
#define CACHE_LINE_SIZE 64

#ifdef PAGE_SIZE
#undef PAGE_SIZE
#endif
#define PAGE_SIZE 0x1000

#ifdef __cache_aligned
#undef __cache_aligned
#endif
#define __cache_aligned __attribute__((__aligned__(CACHE_LINE_SIZE)))

#define ALIGN_CEIL(x,a) (((x)+(a)-1)&~((a)-1))

#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#define MIN(x,y) (((x) > (y)) ? (y) : (x))

#define compiler_fence() __asm__ __volatile__ ("" : : : "memory")

#define memzero(ptr, n) memset((ptr), 0, sizeof(decltype((ptr)[0])) * (n))

namespace nba { namespace knapp {

static inline void insert_pause() {
#ifdef __MIC__
    _mm_delay_32(10);
#else
    __asm volatile ("pause" ::: "memory");
#endif
}

}} // endns(nba::knapp)

#endif //__NBA_KNAPP_MICINTRINSIC_HH__
