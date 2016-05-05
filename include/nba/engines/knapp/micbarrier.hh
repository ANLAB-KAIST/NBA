#ifndef __NBA_KNAPP_MICBARRIER__HH
#define __NBA_KNAPP_MICBARRIER__HH

#ifndef __MIC__
#error "This header should be used by MIC-side codes only."
#endif

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
    uint64_t acc_us;
    uint64_t acc_us_sq;
    uint64_t ts;
    int stat_interval;
    int device_id;
    int num_used;
    int entry_count;
    //bool is_first;
    volatile intptr_t c1 __cache_aligned;
    //volatile intptr_t c2 __attribute__ ((aligned (64)));

public:
    Barrier(int _nThreads, int _device_id, int _stat_interval) :
            nThreads(_nThreads), c1(0), /*c2(0),*/ acc_us(0), acc_us_sq(0),
            num_used(0), ts(0), stat_interval(_stat_interval),
            device_id(_device_id), entry_count(0)
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
        if (iThread) {
            __sync_add_and_fetch((volatile int64_t*)&c1, 1);
            while (c1)
                insert_pause();
        } else {
            //ts = get_usec();
            while ((c1 + 1) != nThreads)
                insert_pause();
            c1 = 0;
            /*
            uint64_t tdiff = get_usec() - ts;
            num_used++;
            acc_us += tdiff;
            acc_us_sq += (tdiff * tdiff);
            if ( num_used % stat_interval == 0 ) {
                double mean_latency = acc_us / (double) num_used;
                double mean_latency_sq = acc_us_sq / (double) num_used;
                double var = mean_latency_sq - (mean_latency * mean_latency);
                fprintf(stderr, "vDevice %d: %dth barrier use (Mean sync latency at %.2lf us, var %.2lf)\n", device_id, num_used, mean_latency, var);
                num_used = 0;
                acc_us = 0;
                acc_us_sq = 0;
            }
            */
        }
    }

};
#else
class Barrier
{
private:
    pthread_spinlock_t lock;
    intptr_t nThreads = 0;
    volatile intptr_t c1;

public:
    Barrier(int _nThreads)
    {
        nThreads = _nThreads;
        c1 = 0;
        pthread_spin_init(&lock, PTHREAD_PROCESS_SHARED);
    }

    void here(intptr_t iThread)
    {
        if (iThread) {
            while (c1)
                insert_pause();
            pthread_spin_lock(&lock);
            __sync_add_and_fetch((volatile int64_t*)&c1, 1);
            pthread_spin_unlock(&lock);
        } else {
            // (iThread==0)
            while ((c1 + 1) != nThreads)
                insert_pause();
            // here when all other threads of team at barrier
            pthread_spin_lock(&lock);
            c1 = 0;
            pthread_spin_unlock(&lock);
        }
    }

};
#endif

}} //endns(nba::knapp)

#endif //__NBA_KNAPP_MICBARRIER__HH

// vim: ts=8 sts=4 sw=4 et
