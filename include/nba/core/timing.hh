#ifndef __NBA_TIMING_HH__
#define __NBA_TIMING_HH__

#include <ctime>
#include <unistd.h>
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

namespace nba {

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
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1e9L + ts.tv_nsec) / 1e3L;
}

}

#endif
