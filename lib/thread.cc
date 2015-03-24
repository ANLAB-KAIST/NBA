/**
 * nShader's Portable Thread Primitives.
 *
 * Authors: Keon Jang <keonjang@an.kaist.ac.kr>,
 *      Joongi Kim <joongi@an.kaist.ac.kr>
 */

#include <cstdio>
#include <sys/time.h>
#include <sys/resource.h>

#include "types.hh"
#include "io.hh"
//#include "computation.hh"
#include "coprocessor.hh"
#include "thread.hh"

#include <rte_config.h>
#include <rte_common.h>
#include <rte_eal.h>


using namespace std;
using namespace nshader;

int nshader::thread_wrapper(void *arg) {
    unsigned core_id = rte_lcore_id();
    const struct thread_collection *const col = (struct thread_collection *) arg;
    unsigned thread_idx = 0;

    unsigned i = 0;
    int found = -1;
    for (i = 0; i < col->num_io_threads; i++) {
        if (core_id == col->io_threads[i].io_ctx->loc.core_id) {
            found = (int)i;
        }
    }

    if (found == -1) {
        /* Corresponding IO thread is not found.
         * Exit silently. */
        return 0;
    }

    return io_loop(col->io_threads[found].io_ctx);
}

unsigned long nshader::get_cpu_idle(int cpu)
{
    unsigned int user = 0, nice = 0, sys = 0;
    unsigned long idle = 0;
    char line[80], cpu_name[32];
    sprintf(cpu_name, "cpu%d", cpu);
    int cpu_name_len = strlen(cpu_name);

    FILE *f = fopen("/proc/stat", "r");
    assert(f != NULL);
    while (fgets(line, sizeof(line), f) != NULL) {
        if (!strncmp(line, cpu_name, cpu_name_len)) {
            sscanf(line + cpu_name_len + 1, "%u %u %u %lu", &user, &nice, &sys, &idle);
            break;
        }
    }
    fclose(f);
    return idle;
}

int nshader::get_thread_time(double *utime, double *stime)
{
    struct rusage ru;
    if (getrusage(RUSAGE_THREAD, &ru) != 0)
        return -1;

    *utime = ru.ru_utime.tv_sec + (ru.ru_utime.tv_usec / 1.0e6);
    *stime = ru.ru_stime.tv_sec + (ru.ru_utime.tv_usec / 1.0e6);
    return 0;
}

// vim: ts=8 sts=4 sw=4 et
