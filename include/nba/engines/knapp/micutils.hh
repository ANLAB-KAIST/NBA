#ifndef __NBA_KNAPP_MICUTILS_HH__
#define __NBA_KNAPP_MICUTILS_HH__

#include <nba/engines/knapp/sharedtypes.hh>
#include <nba/engines/knapp/ctrl.pb.h>
#include <cstdio>
#include <cstdint>
#include <scif.h>
#include <signal.h>

namespace nba {
namespace knapp {

struct vdevice;

static inline void log_error(const char *format, ... )
{
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
}

static inline void log_info(const char *format, ... )
{
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
}

static inline void log_offload(int tid, const char *format, ... )
{
    char str[512];
    va_list args;
    va_start(args, format);
    snprintf(str, 512, "Offload thread %d: %s", tid, format);
    vfprintf(stderr, str, args);
    va_end(args);
}

static inline void log_io(int tid, const char *format, ... )
{
    char str[512];
    va_list args;
    va_start(args, format);
    snprintf(str, 512, "I/O thread %d: %s", tid, format);
    vfprintf(stderr, str, args);
    va_end(args);
}

static inline void log_worker(int tid, const char *format, ... )
{
    char str[512];
    va_list args;
    va_start(args, format);
    snprintf(str, 512, "Worker thread %d: %s", tid, format);
    vfprintf(stderr, str, args);
    va_end(args);
}

static inline void log_device(int vdevice_id, const char *format, ... )
{
    char str[512];
    va_list args;
    va_start(args, format);
    snprintf(str, 512, "vDevice %d: %s", vdevice_id, format);
    vfprintf(stderr, str, args);
    va_end(args);
}

int get_least_utilized_ht(int pcore);

int pollring_init(struct poll_ring *r, int32_t n, scif_epd_t epd);

bool recv_ctrlmsg(scif_epd_t epd, CtrlRequest &req, sigset_t *orig_sigmask = nullptr);

bool send_ctrlresp(scif_epd_t epd, const CtrlResponse &resp);

static inline uint8_t *bufarray_get_va(struct bufarray *ba, int index) {
    if (ba->initialized && index >= 0 && index < (int) ba->size) {
        return ba->bufs[index];
    }
    return nullptr;
}

} // endns(knapp)
} // endns(nba)

#endif //__NBA_KNAPP_MICUTILS_HH__

// vim: ts=8 sts=4 sw=4 et
