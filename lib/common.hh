#ifndef __NSHADER_COMMON_HH__
#define __NSHADER_COMMON_HH__

#define GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)

#include <cstdarg>
#include <cstdio>
#include <cstdint>
#include <cassert>
#include <cstdlib>
#include <ctime>

#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/times.h>
#include <sys/resource.h>


/* Tests on timing precision:
 *
 * getrusage() with RUSAGE_THREAD: about 2 milli-sec
 * clock_gettime(CLOCK_PROCESS_CPUTIME_ID, ...): about 160 nano-sec
 * clock_gettime(CLOCK_THREAD_CPUTIME_ID, ...): about 150 nano-sec
 * clock_gettime(CLOCK_MONOTONIC, ...): about 40 nano-sec
 *
 * Above results are tested with Linux 2.6.36.4-ssl 64bit
 * kernel on Xeon E5650 (2.67 GHz) CPU.
 *
 * See more discussions and benchmark codes:
 * http://stackoverflow.com/questions/6814792/why-is-clock-gettime-so-erratic
 */

#ifdef _POSIX_THREAD_CPUTIME  /* defined in unistd.h */
static inline uint64_t get_thread_cpu_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return ts.tv_sec * 1000000000 + ts.tv_nsec;
}

static inline uint64_t get_thread_cpu_time_unit()
{
    return 1e9L; // nano-seconds
}
#else
static inline uint64_t get_thread_cpu_time()
{
    struct rusage r_;
    getrusage(RUSAGE_THREAD, &r_);

    uint64_t sum = 0;
    sum += r_.ru_utime.tv_sec * 1000000 + r_.ru_utime.tv_usec;
    sum += r_.ru_stime.tv_sec * 1000000 + r_.ru_stime.tv_usec;
    return sum;
}

static inline uint64_t get_thread_cpu_time_unit()
{
    return 1e6L; // micro-seconds
}
#endif

static inline uint64_t get_usec()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (ts.tv_sec * 1e9L + ts.tv_nsec) / 1e3L;
    //struct timeval tv;
    //assert(gettimeofday(&tv, NULL) == 0);
    //return tv.tv_sec * 1000000 + tv.tv_usec;
}

static inline void _cpuid(int i, uint32_t regs[4])
{
#ifdef _WIN32
    __cpuid((int *)regs, (int)i);
#else
    asm volatile
    ("cpuid" : "=a" (regs[0]), "=b" (regs[1]),
           "=c" (regs[2]), "=d" (regs[3])
     : "a" (i), "c" (0));
    // ECX is set to zero for CPUID function 4
#endif
}

/* Intel's documentation suggests use cpuid+rdtsc before meaurements and
 * rdtscp after measurements.  In both cases, we need to add CPUID
 * instruction to prevent out-of-order execution.
 * http://download.intel.com/embedded/software/IA/324264.pdf
 */
static inline uint64_t rdtsc(void)
{
    uint32_t regs[4];
    unsigned low, high;
    _cpuid(0, regs);
    asm volatile("rdtsc" : "=a" (low), "=d" (high));
    return ((uint64_t)low) | (((uint64_t)high) << 32);
}

static inline uint64_t rdtscp(void)
{
    uint32_t low, high;
    uint32_t aux;
    uint32_t regs[4];
    asm volatile ( "rdtscp" : "=a" (low), "=d" (high), "=c" (aux) : : );
    _cpuid(0, regs);
    return ((uint64_t)low | ((uint64_t)high << 32));
}

static inline void set_random(uint8_t *buf, unsigned len)
{
    for (unsigned i = 0; i < len; i++) {
        buf[i] = rand() % 256;
    }
}

#if DEBUG_PRINT
#define DPRINTF printf
#define DPRINT_HEX print_hex
#else
#define DPRINTF(format, ... )
#define DPRINT_HEX(format, ... )
#endif

#define barrier() asm volatile("": : :"memory")

#if 0
#define print_ratelimit(msg, var, nr_iter) do { \
    static uint64_t print_tick_##var; \
    static struct timespec print_prev_ts_##var = {0, 0}; \
    static uint64_t print_ctr_##var; \
    static uint64_t print_min_##var = ULONG_MAX, print_max_##var = 0; \
    print_ctr_##var += (var); \
    if (print_max_##var < (var)) print_max_##var = (var); \
    if (print_min_##var > (var)) print_min_##var = (var); \
    if (print_tick_##var % nr_iter == 0) { \
        struct timespec now_##var; \
        clock_gettime(CLOCK_MONOTONIC_RAW, &now_##var); \
        uint64_t nsec_diff = (1e9L * now_##var.tv_sec + now_##var.tv_nsec) - \
                             (1e9L * print_prev_ts_##var.tv_sec + print_prev_ts_##var.tv_nsec); \
        if (nsec_diff >= 1e9L) { \
            printf("[core:%02u] " msg ": [%lu, %lu]@%lu, %.2f\n", rte_lcore_id(), \
                   print_min_##var, print_max_##var, print_tick_##var, \
                   (float) print_ctr_##var / print_tick_##var); \
            print_ctr_##var = 0; \
            print_tick_##var = 0; \
            print_max_##var = 0; \
            print_min_##var = ULONG_MAX; \
            print_prev_ts_##var = now_##var; \
        } \
    } \
    print_tick_##var ++; \
} while (0)
#endif
#define print_ratelimit(msg, var, nr_iter)

#ifndef CACHE_LINE_SIZE
#  define CACHE_LINE_SIZE 64
#endif
#define __cache_aligned __attribute__((__aligned__(CACHE_LINE_SIZE)))

template<typename T>
static inline T bitselect(int cond, T trueval, T falseval)
{
    int cmp[2] = {0, -1};
    return ((trueval) & cmp[cond]) | ((falseval) & ~cmp[cond]);
}

#define WARN_UNUSED __attribute__((warn_unused_result))

#define ALIGN(x,a) (((x)+(a)-1)&~((a)-1))

typedef unsigned long long ticks;


static inline uint64_t swap64( uint64_t x )
{
    __asm__ ("bswap  %0" : "+r" (x));
    return ( x );
}

static FILE *get_log_file()
{
    char filename[100];
    sprintf(filename, "log_%ld.txt",(long)syscall(SYS_gettid));
    FILE *fp = fopen(filename,"at");
    return fp;
}


static inline void log_printf(const char *fmt, ...)
{
    FILE *fp = get_log_file();
    va_list argptr;
    va_start(argptr, fmt);
    vfprintf(fp, fmt, argptr);
    va_end(argptr);
    fclose(fp);
}


static inline void fprint_hex(FILE *fp, const uint8_t *binary, const int length) {
    for(int i = 0; i < length; i++){
        fprintf(fp, "0x%02hhx,",binary[i]);
        if (i % 16 == 15)
            fprintf(fp, "\n");

    }
    fprintf(fp, "\n");
}

static inline void log_print_hex(const uint8_t *binary, const int length)
{
    FILE* fp = get_log_file();
    fprint_hex(fp, binary, length);
    fclose(fp);
}

static inline void print_hex(const uint8_t *binary, const int length) {
    fprint_hex(stderr, binary, length);
}

#endif

// vim: ts=8 sts=4 sw=4 et
