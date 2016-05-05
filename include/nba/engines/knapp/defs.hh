#ifndef __NBA_KNAPP_DEFS_HH__
#define __NBA_KNAPP_DEFS_HH__

/* Behavioral limits. */
#define KNAPP_MAX_KERNEL_ARGS (16)
#define KNAPP_SCIF_MAX_CONN_RETRY (5)
#define KNAPP_OFFLOAD_CTRLBUF_SIZE (32)
#define KNAPP_OFFLOAD_COMPLETE (0xdeadbeefull)
#define KNAPP_TASK_READY (0xcafebabeull)
#define KNAPP_COPY_PENDING (~((uint64_t)0))
#define KNAPP_VDEV_PROFILE_INTERVAL 1000
#define KNAPP_BARRIER_PROFILE_INTERVAL 100

/* Hardware limits. */
#define KNAPP_THREADS_LIMIT (240)
#define KNAPP_NUM_INT32_PER_VECTOR (16)
#define KNAPP_NUM_CORES (60)
#define KNAPP_MAX_THREADS_PER_CORE (4)
#define KNAPP_MAX_CORES_PER_DEVICE (60)
#define KNAPP_MAX_LCORES_PER_DEVICE (240)

/* Base numbers. */
#define KNAPP_MASTER_PORT (3000)
#define KNAPP_HOST_DATA_PORT_BASE (2000)
#define KNAPP_HOST_CTRL_PORT_BASE (2100)
#define KNAPP_MIC_DATA_PORT_BASE (2200)
#define KNAPP_MIC_CTRL_PORT_BASE (2300)


namespace nba { namespace knapp {

inline int mic_pcore_to_lcore(int pcore, int ht) {
    return (pcore * KNAPP_MAX_THREADS_PER_CORE + ht + 1)
           % (KNAPP_NUM_CORES * KNAPP_MAX_THREADS_PER_CORE);
}

}} // endns(nba::knapp)


#endif //__NBA_KNAPP_DEFS_HH__

// vim: ts=8 sts=4 sw=4 et
