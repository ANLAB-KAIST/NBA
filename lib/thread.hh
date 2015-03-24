#ifndef __NBA_THREAD_HH__
#define __NBA_THREAD_HH__

/**
 * NBA's Portable Thread Primitives.
 *
 * Authors: Keon Jang <keonjang@an.kaist.ac.kr>,
 *      Joongi Kim <joongi@an.kaist.ac.kr>
 */

#include "config.hh"
#include "types.hh"

#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cassert>
extern "C" {
#include <sched.h>
#include <numa.h>
#include <pthread.h>
#include <signal.h>
#include <syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <sys/eventfd.h>
#include <fcntl.h>
#include <unistd.h>
#include <rte_branch_prediction.h>
}

namespace nba
{
    typedef pthread_t thread_id_t;

    int thread_wrapper(void *arg);

    class Lock
    {
    public:
        Lock()
        {
            int ret;
            ret = pthread_mutex_init(&mutex_, NULL);
            assert(ret == 0);
        }

        void acquire()
        {
            pthread_mutex_lock(&mutex_);
        }

        void release()
        {
            pthread_mutex_unlock(&mutex_);
        }

        pthread_mutex_t mutex_;
    } __attribute__ ((aligned (64)));

    class CondVar
    {
    public:
        CondVar()
        {
            int ret;
            ret = pthread_cond_init(&cond_, NULL);
            assert(ret == 0);
            ret = pthread_mutex_init(&mutex_, NULL);
            assert(ret == 0);
        }
        virtual ~CondVar()
        {
            pthread_cond_destroy(&cond_);
            pthread_mutex_destroy(&mutex_);
        }

        void lock()
        {
            pthread_mutex_lock(&mutex_);
        }

        void unlock()
        {
            pthread_mutex_unlock(&mutex_);
        }

        void wait()
        {
            pthread_cond_wait(&cond_, &mutex_);
        }

        void signal()
        {
            pthread_cond_signal(&cond_);
        }

        void signal_all()
        {
            pthread_cond_broadcast(&cond_);
        }

    private:
        pthread_cond_t cond_;
        pthread_mutex_t mutex_;
    } __attribute__ ((aligned (64)));

    class UserEvent
    {
    public:
        UserEvent() : evfd_(-1), is_poll_mode_(false)
        {
            evfd_ = eventfd(0, 0);
            assert(evfd_ > 0);
        }

        virtual ~UserEvent()
        {
            close(evfd_);
        }

        void trigger()
        {
            uint64_t v = 1;
            write(evfd_, &v, sizeof(v));
        }

        void trigger(uint64_t v)
        {
            write(evfd_, &v, sizeof(v));
        }

        void wait()
        {
            uint64_t v;
            if (unlikely(is_poll_mode_)) {
                long flags;
                flags = fcntl(evfd_, F_GETFL);
                flags &= ~O_NONBLOCK;
                fcntl(evfd_, F_SETFL, flags);
                is_poll_mode_ = false;
            }
            read(evfd_, &v, sizeof(v));  // This will block until trigger() is called.
        }

        bool is_triggered()
        {
            int ret;
            uint64_t v;
            if (unlikely(!is_poll_mode_)) {
                long flags;
                flags = fcntl(evfd_, F_GETFL);
                flags |= O_NONBLOCK;
                fcntl(evfd_, F_SETFL, flags);
                is_poll_mode_ = true;
            }
            ret = read(evfd_, &v, sizeof(v));
            return !(ret <= 0 && errno == EAGAIN);
        }

        int getfd()
        {
            return evfd_;
        }

    private:
        int evfd_;
        bool is_poll_mode_;
    } __attribute__ ((aligned (64)));

    class AsyncSemaphore
    {
        /* It would be better to use plain semaphore,
         * but it does not provide notification on when it becomes
         * available. :( */
    public:
        AsyncSemaphore(unsigned int initval)
        {
            evfd_ = eventfd(initval, EFD_SEMAPHORE | EFD_NONBLOCK);
            assert(evfd_ > 0);
        }

        virtual ~AsyncSemaphore()
        {
            close(evfd_);
        }

        void up(uint64_t count = 1)
        {
            write(evfd_, &count, sizeof(count));
        }

        uint64_t try_down()
        {
            uint64_t v = 0;
            int ret;
            ret = read(evfd_, &v, sizeof(v));
            if (ret == -1 && errno == EAGAIN)
                return 0;
            return v;
        }

        int getfd()
        {
            return evfd_;
        }
    private:
        int evfd_;
    } __attribute__ ((aligned(64)));

    class CountedBarrier
    {
    public:
        CountedBarrier(unsigned count)
        {
            _count = count;
            _current = 0;
        }

        void reset()
        {
            _condvar.lock();
            _current = 0;
            _condvar.unlock();
        }

        void wait()
        {
            _condvar.lock();
            if (_current < _count)
                _condvar.wait();
            _condvar.unlock();
        }

        void proceed()
        {
            _condvar.lock();
            _current ++;
            if (_current >= _count)
                _condvar.signal();
            _condvar.unlock();
        }

    private:
        unsigned _count;
        unsigned _current;
        CondVar _condvar;
    } __attribute__ ((aligned (64)));

    class EventChannel
    {
    public:
        EventChannel()
        {
            assert(0 == pipe(pipefd));
        }

        virtual ~EventChannel()
        {
            close(pipefd[0]);
            close(pipefd[1]);
        }

        void send(uint64_t v)
        {
            write(pipefd[1], &v, sizeof(v));
        }

        uint64_t receive()
        {
            uint64_t v;
            read(pipefd[0], &v, sizeof(v));  // This will block until send() is called.
            return v;
        }

        int getfd()
        {
            return pipefd[0];
        }

    private:
        int pipefd[2];
    } __attribute__ ((aligned (64)));

    template<class T>
    class Thread
    {
    public:
        Thread(int core_id, T *r)
        {
            core_id_ = core_id;
            numa_node_id_ = numa_node_of_cpu(core_id_);
            r_ = r;
        }

        void start()
        {
            int s = pthread_attr_init(&attr_);
            if (s != 0) {
                fprintf(stderr, "pthread_attr_init failed\n");
                exit(-1);
            }

            s = pthread_create(&thread_, &attr_,
                       &Thread<T>::thread_func,
                       (void*)this);
            if (s != 0) {
                fprintf(stderr, "thread_create failed\n");
                exit(-1);
            }
        }
        void stop()
        {
            r_->stop();
            pthread_kill(thread_, SIGTERM);
        }

        void join()
        {
            pthread_join(thread_, NULL);
        }

        void set_priority(int p)
        {
            pid_t tid = syscall(SYS_gettid);
            setpriority(PRIO_PROCESS, tid, p);
        }

        static void *thread_func(void *obj)
        {
            Thread* t = (Thread*)obj;
            bind_cpu(t->core_id_);
            t->r_->run();
            return NULL;
        }

        static int get_num_cpus()
        {
            return sysconf(_SC_NPROCESSORS_ONLN);
        }

        static int bind_cpu(int cpu)
        {
            struct bitmask *bmask;
            int ncpus = numa_num_configured_cpus();

            bmask = numa_bitmask_alloc(ncpus);
            assert(bmask != NULL);
            assert(cpu >= 0 && cpu < ncpus);
            numa_bitmask_clearall(bmask);
            numa_bitmask_setbit(bmask, cpu);
            numa_sched_setaffinity(0, bmask);
            numa_bitmask_free(bmask);

            /* Reference:
               On dual Sandybridge (E5-2670) systems,
                 cpu 0 to 7: node 0
                 cpu 8 to 15: node 1
                 cpu 16 to 23: node 0 (ht)
                 cpu 24 to 31: node 1 (ht)
               For legacy-ordered MADT (multiple APIC description table) systems,
                 cpu even: node 0
                 cpu odd: node 1
             */
            bmask = numa_bitmask_alloc(numa_num_configured_nodes());
            assert(bmask != NULL);
            numa_bitmask_clearall(bmask);
            numa_bitmask_setbit(bmask, numa_node_of_cpu(cpu));
            numa_set_membind(bmask);
            numa_bitmask_free(bmask);

            return 0;
        }

        int get_core()
        {
            return core_id_;
        }

        int get_numa_node()
        {
            return numa_node_id_;
        }

        static void yield()
        {
            pthread_yield();
        }

        static thread_id_t self()
        {
            return (thread_id_t) pthread_self();
        }

        static bool is_thread_equal(thread_id_t a, thread_id_t b)
        {
            return pthread_equal(a, b);
        }

    private:
        T *r_;
        pthread_t thread_;
        pthread_attr_t attr_;
        int core_id_;
        int numa_node_id_;

    } __attribute__ ((aligned (64)));

    class Runnable
    {
    public:
        Runnable(){};
        ~Runnable(){};
        virtual void run() = 0;
        virtual void stop() = 0;
        int get_numa_node()
        {
            return thread_->get_numa_node();
        }
        int get_core()
        {
            return thread_->get_core();
        }
        void set_priority(int p)
        {
            thread_->set_priority(p);
        }
    protected:
        Thread<Runnable> *thread_;
    };

    typedef Thread<Runnable> threading;

    unsigned long get_cpu_idle(int cpu);
    int get_thread_time(double *utime, double *stime);
}

#endif /* __THREAD_HH__ */

// vim: ts=8 sts=4 sw=4 et
