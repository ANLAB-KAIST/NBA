#ifndef __NBA_KNAPP_DEFS_HH__
#define __NBA_KNAPP_DEFS_HH__

#include <cstdint>
#include <tuple>

/* Behavioral limits. */
#define KNAPP_MAX_KERNEL_ARGS (16)
#define KNAPP_SYNC_CYCLES (1000000u)
#define KNAPP_VDEV_PROFILE_INTERVAL 1000
#define KNAPP_BARRIER_PROFILE_INTERVAL 100

/* Software limits. */
#define KNAPP_VDEV_MAX_POLLRINGS (8)
// 4bits: iobase_id (==task_id)
// 1bit:  input or output
#define KNAPP_VDEV_MAX_RMABUFFERS (32)
#define KNAPP_GLOBAL_MAX_RMABUFFERS (64)

/* Hardware limits. */
#define KNAPP_THREADS_LIMIT (240)
#define KNAPP_NUM_INT32_PER_VECTOR (16)
#define KNAPP_NUM_CORES (60)
#define KNAPP_MAX_THREADS_PER_CORE (4)
#define KNAPP_MAX_CORES_PER_DEVICE (60)
#define KNAPP_MAX_LCORES_PER_DEVICE (240)

/* Base numbers. */
#define KNAPP_CTRL_PORT (3000)
#define KNAPP_HOST_DATA_PORT_BASE (2000)
#define KNAPP_HOST_CTRL_PORT_BASE (2100)
#define KNAPP_MIC_DATA_PORT_BASE (2200)
#define KNAPP_MIC_CTRL_PORT_BASE (2300)


namespace nba { namespace knapp {

enum poll_ring_state : uint64_t {
    KNAPP_H2D_COMPLETE = 0xdeadbeefull,
    KNAPP_D2H_COMPLETE = 0xeadbeedfull,
    KNAPP_TASK_READY = 0xcafebabeull,
    KNAPP_COPY_PENDING = (~(0ull)),
    KNAPP_TERMINATE = 0xfff0f0cccull
};

enum rma_direction : uint32_t {
    INPUT = 0,
    OUTPUT = 1
};

inline int mic_pcore_to_lcore(int pcore, int ht) {
    return (pcore * KNAPP_MAX_THREADS_PER_CORE + ht + 1)
           % (KNAPP_NUM_CORES * KNAPP_MAX_THREADS_PER_CORE);
}

/* Special buffers. */
const uint32_t BUFFER_TASK_PARAMS = 0x80000001u;
const uint32_t BUFFER_D2H_PARAMS = 0x80000002u;

/* Buffer ID = 4 bits task ID + 1 bit direction (input or output) */
constexpr uint32_t BUFFER_GLOBAL_FLAG    = 0x20u;
constexpr uint32_t BUFFER_TASKID_MASK    = 0x1eu;
constexpr uint32_t BUFFER_TASKID_SHIFT   = 1u;
constexpr uint32_t BUFFER_DIRECTION_MASK = 0x01u;

static_assert((BUFFER_GLOBAL_FLAG | BUFFER_TASKID_MASK | BUFFER_DIRECTION_MASK) <= KNAPP_GLOBAL_MAX_RMABUFFERS,
        "KNAPP_GLOBAL_MAX_RMABUFFERS cannot represent all possible buffer IDs.");

constexpr uint32_t compose_buffer_id(bool is_global, uint32_t task_id, rma_direction dir)
{
    return (is_global ? BUFFER_GLOBAL_FLAG : 0u)
           | ((task_id << BUFFER_TASKID_SHIFT) & BUFFER_TASKID_MASK)
           | static_cast<uint32_t>(dir);
}

constexpr bool is_global_buffer(uint32_t buffer_id)
{
    return (buffer_id & BUFFER_GLOBAL_FLAG) != 0;
}

constexpr uint32_t to_task_id(uint32_t buffer_id)
{
    return (buffer_id & BUFFER_TASKID_MASK) >> BUFFER_TASKID_SHIFT;
}

constexpr rma_direction to_direction(uint32_t buffer_id)
{
    return static_cast<rma_direction>(buffer_id & BUFFER_DIRECTION_MASK);
}

const inline std::tuple<bool, uint32_t, rma_direction>
decompose_buffer_id(uint32_t buffer_id)
{
    bool is_global    = is_global_buffer(buffer_id);
    uint32_t task_id  = to_task_id(buffer_id);
    rma_direction dir = to_direction(buffer_id);
    return std::make_tuple(is_global, task_id, dir);
}

}} // endns(nba::knapp)


#endif //__NBA_KNAPP_DEFS_HH__

// vim: ts=8 sts=4 sw=4 et
