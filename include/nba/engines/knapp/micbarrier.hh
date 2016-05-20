#ifndef __NBA_KNAPP_MICBARRIER__HH
#define __NBA_KNAPP_MICBARRIER__HH

#ifndef __MIC__
#error "This header should be used by MIC-side codes only."
#endif

#include <nba/engines/knapp/defs.hh>
#include <nba/engines/knapp/micintrinsic.hh>
#include <cstdint>
#include <pthread.h>
#include <time.h>

#define _USE_ATOMIC_BARRIER_

namespace nba { namespace knapp {

#ifdef _USE_ATOMIC_BARRIER_
class Barrier
{
private:
    intptr_t nThreads;
    int stat_interval;
    int device_id;

    volatile intptr_t c1 __cache_aligned;

public:
    Barrier(int _nThreads, int _device_id, int _stat_interval) :
            nThreads(_nThreads), c1(0),
            device_id(_device_id)
    { }

    virtual ~Barrier()
    { }

    uint64_t get_usec(void)
    {
        struct timespec now;
        //clock_gettime(CLOCK_MONOTONIC_RAW, &now);
        clock_gettime(CLOCK_MONOTONIC, &now);
        return now.tv_sec * 1000000L + now.tv_nsec / 1000L;
    }

    void here(intptr_t iThread)
    {
        uint32_t count = 0;
        if (iThread) {
            __sync_add_and_fetch((volatile int64_t*)&c1, 1);
            while (c1) {
                insert_pause();
            }
        } else {
            //ts = get_usec();
            while ((c1 + 1) != nThreads) {
                insert_pause();
            }
            c1 = 0;
        }
    }

};
#else
class Barrier
{
private:
    pthread_barrier_t b;
    int _num_threads;

public:
    Barrier(int num_threads, int device_id, int interval) : _num_threads(num_threads)
    {
        pthread_barrier_init(&b, nullptr, num_threads);
    }

    virtual ~Barrier()
    {
        pthread_barrier_destroy(&b);
    }

    void here(int cur_thread)
    {
        pthread_barrier_wait(&b);
    }

};
#endif

}} //endns(nba::knapp)

#endif //__NBA_KNAPP_MICBARRIER__HH

// vim: ts=8 sts=4 sw=4 et
