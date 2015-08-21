#ifndef __NBA_COMMON_HH__
#define __NBA_COMMON_HH__

#define GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)

#include <cstdarg>
#include <cstdio>
#include <cstdint>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <unistd.h>
#include <sys/syscall.h>

#if DEBUG_PRINT
#define DPRINTF printf
#define DPRINT_HEX print_hex
#else
#define DPRINTF(format, ... )
#define DPRINT_HEX(format, ... )
#endif

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

#define WARN_UNUSED __attribute__((warn_unused_result))

#define ALIGN(x,a) (((x)+(a)-1)&~((a)-1))

typedef unsigned long long ticks;


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
