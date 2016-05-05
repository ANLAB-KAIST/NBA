#include <nba/engines/knapp/defs.hh>
#include <nba/engines/knapp/mictypes.hh>
#include <nba/engines/knapp/sharedtypes.hh>
#include <nba/engines/knapp/micbarrier.hh>
#include <nba/engines/knapp/micutils.hh>
#include <nba/engines/knapp/ctrl.pb.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <unistd.h>
#include <poll.h>
#include <signal.h>
#include <locale.h>

namespace nba { namespace knapp {

extern char **kernel_paths;
extern char **kernel_names;
std::vector<struct nba::knapp::vdevice *> vdevs;
extern worker_func_t worker_funcs[];

/* MIC daemon consists of 3 types of threads:
 * (1) control_thread_loop: global state mgmt (e.g., "cudaMalloc", "cudaMmecpy")
 * (2) master_thread_loop: vDevice controller
 * (3) worker_thread_loop: vDevice worker
 *
 * Each vDevice has at least one master and zero or more workers.
 */

static uint32_t mic_num_pcores = 0;
static uint32_t mic_num_lcores = 0;
static uint64_t global_vdevice_counter = 0;
static bool pcore_used[KNAPP_NUM_CORES];
static pthread_t control_thread;
static volatile bool exit_flag = false;

void *control_thread_loop(void *arg);
void *master_thread_loop(void *arg);
void *worker_thread_loop(void *arg);
void stop_all();
void handle_signal(int signal);

struct vdevice *create_vdev(
        uint32_t num_pcores,
        uint32_t num_lcores_per_pcore,
        uint32_t pipeline_depth);
void destroy_vdev(struct vdevice *vdev);

}} // endns(nba::knapp)

using namespace nba;
using namespace nba::knapp;

static struct vdevice *nba::knapp::create_vdev(
        uint32_t num_pcores,
        uint32_t num_lcores_per_pcore,
        uint32_t pipeline_depth)
{
    int rc;
    bool avail = true;
    uint32_t pcore_begin = 0;
    struct vdevice *vdev = nullptr;

    /* Find available slots and allocate among MIC cores. */
    for (uint32_t i = 0; i < mic_num_pcores - num_pcores; i++) {
        for (uint32_t j = i; j < num_pcores; j++) {
            if (pcore_used[j]) {
                avail = false;
                break;
            }
        }
        if (avail) {
            pcore_begin = i;
            break;
        }
    }
    if (avail) {
        vdev = new struct vdevice();
        memzero(vdev, 1);
        vdev->pcores.clear();
        vdev->lcores.clear();
        for (uint32_t i = 0; i < num_pcores; i++) {
            pcore_used[pcore_begin + i] = true;
            vdev->pcores.push_back(pcore_begin + i);
            for (uint32_t j = 0; j < num_lcores_per_pcore; j++) {
                vdev->lcores.push_back(mic_pcore_to_lcore(pcore_begin + i, j));
            }
        }
    } else {
        return nullptr;
    }
    vdev->device_id = (++global_vdevice_counter);
    vdev->pipeline_depth = pipeline_depth;
    vdev->ht_per_core = num_lcores_per_pcore;

    /* Initialize barriers. */
    vdev->data_ready_barriers = (Barrier **) _mm_malloc(sizeof(Barrier *) * vdev->pipeline_depth, CACHE_LINE_SIZE);
    vdev->task_done_barriers  = (Barrier **) _mm_malloc(sizeof(Barrier *) * vdev->pipeline_depth, CACHE_LINE_SIZE);
    for (uint32_t i = 0; i < vdev->pipeline_depth; i++) {
        vdev->data_ready_barriers[i] = new Barrier(vdev->num_worker_threads, vdev->device_id, KNAPP_BARRIER_PROFILE_INTERVAL);
        vdev->task_done_barriers[i]  = new Barrier(vdev->num_worker_threads, vdev->device_id, KNAPP_BARRIER_PROFILE_INTERVAL);
    }

    //recv_ctrlmsg(vdev->ctrl_epd, vdev->ctrlbuf, OP_MALLOC, &vdev->inputbuf_size, &vdev->resultbuf_size, &vdev->pipeline_depth, &vdev->remote_writebuf_base_ra);
    //assert ( 0 == bufarray_ra_init(&vdev->inputbuf_array, vdev->pipeline_depth, vdev->inputbuf_size, PAGE_SIZE, vdev->data_epd, SCIF_PROT_READ | SCIF_PROT_WRITE) );
    //assert ( 0 == bufarray_ra_init(&vdev->resultbuf_array, vdev->pipeline_depth, vdev->resultbuf_size, PAGE_SIZE, vdev->data_epd, SCIF_PROT_READ | SCIF_PROT_WRITE) );

#if 0
    /* Initialize pollring. */
    // TODO: init vdev->data_epd
    int32_t pollring_len = vdev->pipeline_depth;
    rc = pollring_init(&vdev->poll_ring, pollring_len, vdev->data_epd);
    assert(0 == rc);
    uint64_t volatile *pollring = vdev->poll_ring.ring;
    for (int i = 0; i < pollring_len; i++) {
        pollring[i] = KNAPP_COPY_PENDING;
    }
    //send_ctrlresp(vdev->ctrl_epd, vdev->ctrlbuf, OP_REG_POLLRING, &vdev->poll_ring.ring_ra, NULL, NULL, NULL);

    /* Initialize input buffers. */
    // TODO: implement

    /* Initialize output buffers. */
    // TODO: implement
#endif
    return vdev;
}

static void nba::knapp::destroy_vdev(struct vdevice *vdev)
{
    // TODO: destroy threads.
    for (uint32_t i = 0; i < vdev->pipeline_depth; i++) {
        delete vdev->data_ready_barriers[i];
        delete vdev->task_done_barriers[i];
    }
    _mm_free(vdev->data_ready_barriers);
    _mm_free(vdev->task_done_barriers);
    delete vdev;
}


static void *nba::knapp::worker_thread_loop(void *arg)
{
    struct worker_thread_info *info = (struct worker_thread_info *) arg;
    int tid = info->thread_id;
    struct vdevice *vdev = info->vdev;

    // TODO: retrieve from per-task info
    worker_func_t pktproc_func = info->pktproc_func;

    while (!vdev->exit) {
        //worker_preproc(tid, vdev);
        int task_id = vdev->cur_task_id;
        struct worker *w = &vdev->per_thread_work_info[task_id][tid];
        //pktproc_func(w);
        //worker_postproc(tid, vdev);
    }
    return nullptr;
}

static void *nba::knapp::master_thread_loop(void *arg)
{
    struct vdevice *vdev = (struct vdevice *) arg;
    /* At this point, vdev is already initialized completely. */

    int backlog = 16;
    int rc = 0;
    log_device(vdev->device_id, "Listening on control/data channels...\n");

    rc = scif_listen(vdev->data_listen_epd, backlog);
    assert(0 == rc);
    rc == scif_listen(vdev->ctrl_listen_epd, backlog);
    assert(0 == rc);
    rc = scif_accept(vdev->data_listen_epd, &vdev->remote_data_port,
                     &vdev->data_epd, SCIF_ACCEPT_SYNC);
    assert(0 == rc);
    log_device(vdev->device_id, "Connection established between "
               "local dataport (%d, %d) and remote dataport (%d, %d)\n",
               vdev->local_data_port.node, vdev->local_data_port.port,
               vdev->remote_data_port.node, vdev->remote_data_port.port);
    rc = scif_accept(vdev->ctrl_listen_epd, &vdev->remote_ctrl_port,
                     &vdev->ctrl_epd, SCIF_ACCEPT_SYNC);
    assert(0 == rc);
    log_device(vdev->device_id, "Connection established between "
               "local ctrlport (%d, %d) and remote ctrlport (%d, %d)\n",
               vdev->local_ctrl_port.node, vdev->local_ctrl_port.port,
               vdev->remote_ctrl_port.node, vdev->remote_ctrl_port.port);

    vdev->worker_threads = (pthread_t *) _mm_malloc(sizeof(pthread_t) * vdev->num_worker_threads, CACHE_LINE_SIZE);

    assert ( vdev->worker_threads != NULL );
	vdev->thread_info_array = (struct worker_thread_info *) _mm_malloc(sizeof(struct worker_thread_info) * vdev->num_worker_threads, CACHE_LINE_SIZE);
	assert ( vdev->thread_info_array != NULL );
	for ( int i = 0; i < vdev->num_worker_threads; i++ ) {
		struct worker_thread_info *info = &vdev->thread_info_array[i];
		info->thread_id = i;
		info->vdev = vdev;
		info->pktproc_func = vdev->worker_func;
	}
    vdev->per_thread_work_info = (struct worker **) _mm_malloc(sizeof(struct worker *) * vdev->pipeline_depth, CACHE_LINE_SIZE);
    assert ( vdev->per_thread_work_info != NULL );
    for ( int i = 0; i < (int) vdev->pipeline_depth; i++ ) {
        vdev->per_thread_work_info[i] = (struct worker *) _mm_malloc(sizeof(struct worker) * vdev->num_worker_threads, CACHE_LINE_SIZE);
        assert ( vdev->per_thread_work_info[i] != NULL );
    }
    log_device(vdev->device_id, "Allocated %d per-worker-thread work info\n", vdev->num_worker_threads);

    for ( int pdepth = 0; pdepth < (int) vdev->pipeline_depth; pdepth++ ) {
        for ( unsigned ithread = 0; ithread < vdev->num_worker_threads; ithread++ ) {
            //init_worker(&vdev->per_thread_work_info[pdepth][ithread], ithread, vdev->workload_type, vdev, pdepth);
        }
    }
    for ( unsigned i = 0; i < vdev->lcores.size(); i++ ) {
        //log_device(vdev->device_id, "Creating thread for lcore %d (%d, %d) and thread %d\n", lcore, pcore, ht, ithread);
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        size_t cpuset_sz = CPU_ALLOC_SIZE(vdev->lcores.size());
        cpu_set_t *cpuset = CPU_ALLOC(vdev->lcores.size());
        CPU_ZERO_S(cpuset_sz, cpuset);
        CPU_SET_S(vdev->lcores[i], cpuset_sz, cpuset); // pin to the first core.
        pthread_attr_setaffinity_np(&attr, cpuset_sz, cpuset);
        CPU_FREE(cpuset);
        rc = pthread_create(&vdev->worker_threads[i], &attr,
                            worker_thread_loop,
                            (void *) &vdev->thread_info_array[i]);
    }

    log_device(vdev->device_id, "Running processing daemon...\n");
    int32_t cur_task_id = 0;
    uint64_t volatile *pollring = vdev->poll_ring.ring;
    while (true) {
        // TODO: State safety check?
        while (pollring[cur_task_id] != KNAPP_OFFLOAD_COMPLETE) {
            insert_pause();
        }
        compiler_fence();
        uint8_t *inputbuf_va = bufarray_get_va(&vdev->inputbuf_array, cur_task_id);
        //uint8_t *resultbuf_va = bufarray_get_va(&vdev->resultbuf_array, cur_task_id);
        struct taskitem *ti = (struct taskitem *) inputbuf_va;

        if ( ti->task_id != cur_task_id ) {
            log_device(vdev->device_id, "offloaded task id (%d) doesn't match pending id (%d)\n", ti->task_id, cur_task_id);
            exit(1);
        }
        //log_device(vdev->device_id, "Queuing task: id (%d), input_size (%d), num_packets (%d)\n", ti->task_id, (int) ti->input_size, ti->num_packets);
        uint8_t *inputbuf_payload_va = (uint8_t *)(ti + 1);
        uint64_t input_size = ti->input_size;
        int32_t num_packets = ti->num_packets;
        vdev->num_packets_in_cur_task = num_packets;
        int32_t to_process = num_packets;
        for ( int ithread = 0; ithread < vdev->num_worker_threads; ithread++ ) {
            struct worker *w = &vdev->per_thread_work_info[cur_task_id][ithread];
            w->num_packets = MIN(to_process, w->max_num_packets);
            to_process -= w->num_packets;
        }
        compiler_fence();
        pollring[cur_task_id] = KNAPP_TASK_READY;

        // FIXME: work the next line into the stats somehow
        // total_packets_processed += num_packets;
        if ( vdev->exit == true ) {
            for ( int i = 0; i < vdev->num_worker_threads; i++ ) {
                for ( int pdepth = 0; pdepth < vdev->pipeline_depth; pdepth++ ) {
                    vdev->per_thread_work_info[pdepth][i].exit = true;
                }
            }
            break;
        }
        cur_task_id = (cur_task_id + 1) % (vdev->poll_ring.len);
    }
    for ( int i = 0; i < vdev->num_worker_threads; i++ ) {
        pthread_join(vdev->worker_threads[i], NULL);
    }
    scif_close(vdev->data_epd);
    scif_close(vdev->data_listen_epd);
    scif_close(vdev->ctrl_epd);
    scif_close(vdev->ctrl_listen_epd);
    return nullptr;
}

static void *nba::knapp::control_thread_loop(void *arg)
{
    scif_epd_t master_listen_epd, master_epd;
    struct scif_portID accepted_master_port;
    int backlog = 1;
    int rc = 0;
    sigset_t intr_mask, orig_mask;
    sigemptyset(&intr_mask);
    sigaddset(&intr_mask, SIGINT);
    sigaddset(&intr_mask, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &intr_mask, &orig_mask);

    uint16_t scif_nodes[32];
    uint16_t local_node;
    size_t num_nodes;
    num_nodes = scif_get_nodeIDs(scif_nodes, 32, &local_node);

    master_listen_epd = scif_open();
    rc = scif_bind(master_listen_epd, KNAPP_MASTER_PORT);
    assert(KNAPP_MASTER_PORT == rc);
    rc = scif_listen(master_listen_epd, backlog);
    assert(0 == rc);

    log_info("Starting the control channel...\n");
    /* For simplicty, we allow only a single concurrent connection. */
    while (!exit_flag) {
        struct pollfd p = {master_listen_epd, POLLIN, 0};
        rc = ppoll(&p, 1, nullptr, &orig_mask);
        if (rc == -1 && errno == EINTR && exit_flag)
            break;
        rc = scif_accept(master_listen_epd, &accepted_master_port,
                         &master_epd, SCIF_ACCEPT_SYNC);
        assert(0 == rc);

        log_info("A control session started.\n");
        CtrlRequest request;
        CtrlResponse resp;

        while (!exit_flag) {
            resp.Clear();
            if (!recv_ctrlmsg(master_epd, request, &orig_mask))
                // usually, EINTR or ECONNRESET.
                break;
            switch (request.type()) {
            case CtrlRequest::PING:
                if (request.has_text()) {
                    const std::string &msg = request.text().msg();
                    log_info("CONTROL: PING with \"%s\"\n", msg.c_str());
                    resp.set_reply(CtrlResponse::SUCCESS);
                    resp.mutable_text()->set_msg(msg);
                } else {
                    resp.set_reply(CtrlResponse::INVALID);
                    resp.mutable_text()->set_msg("Invalid parameter.");
                }
                break;
            case CtrlRequest::MALLOC:
                if (request.has_malloc()) {
                    void *ptr = _mm_malloc(request.malloc().size(), request.malloc().align());
                    if (ptr == nullptr) {
                        resp.set_reply(CtrlResponse::FAILURE);
                        resp.mutable_text()->set_msg("_mm_malloc failed.");
                    } else {
                        resp.set_reply(CtrlResponse::SUCCESS);
                        resp.mutable_resource()->set_handle((uintptr_t) ptr);
                    }
                } else {
                    resp.set_reply(CtrlResponse::INVALID);
                    resp.mutable_text()->set_msg("Invalid parameter.");
                }
                break;
            case CtrlRequest::FREE:
                if (request.has_resource()) {
                    void *ptr = (void *) request.resource().handle();
                    _mm_free(ptr);
                    resp.set_reply(CtrlResponse::SUCCESS);
                } else {
                    resp.set_reply(CtrlResponse::INVALID);
                    resp.mutable_text()->set_msg("Invalid parameter.");
                }
                break;
            case CtrlRequest::CREATE_VDEV:
                if (request.has_vdevinfo()) {
                    struct vdevice *vdev = create_vdev(request.vdevinfo().num_pcores(),
                                                       request.vdevinfo().num_lcores_per_pcore(),
                                                       request.vdevinfo().pipeline_depth());
                    if (vdev == nullptr) {
                        resp.set_reply(CtrlResponse::FAILURE);
                        resp.mutable_text()->set_msg("vDevice creation failed.");
                    } else {
                        resp.set_reply(CtrlResponse::SUCCESS);
                        resp.mutable_resource()->set_handle((uintptr_t) vdev);
                    }
                } else {
                    resp.set_reply(CtrlResponse::INVALID);
                    resp.mutable_text()->set_msg("Invalid parameter.");
                }
                break;
            case CtrlRequest::DESTROY_VDEV:
                if (request.has_resource()) {
                    struct vdevice *vdev = (struct vdevice *) request.resource().handle();
                    destroy_vdev(vdev);
                    resp.set_reply(CtrlResponse::SUCCESS);
                } else {
                    resp.set_reply(CtrlResponse::INVALID);
                    resp.mutable_text()->set_msg("Invalid parameter.");
                }
                break;
            default:
                log_error("CONTROL: Not implemented request type: %d\n", request.type());
                resp.set_reply(CtrlResponse::INVALID);
                resp.mutable_text()->set_msg("Invalid request type.");
                break;
            }
            if (!send_ctrlresp(master_epd, resp))
                break;
        }
        scif_close(master_epd);
        log_info("The control session terminated.\n");
    }
    log_info("Terminating the control channel...\n");
    scif_close(master_listen_epd);

    return nullptr;
}


static void nba::knapp::stop_all() {
    exit_flag = true;
    for (auto vdev : vdevs)
        vdev->exit = true;

    log_info("Stopping...\n");

    /* Ensure propagation of signals. */
    pthread_kill(control_thread, SIGINT);
    for (auto vdev : vdevs)
        pthread_kill(vdev->master_thread, SIGINT);

    /* Wait until all finishes. */
    for (auto vdev : vdevs)
        pthread_join(vdev->master_thread, nullptr);
    pthread_join(control_thread, nullptr);
}

static void nba::knapp::handle_signal(int signal) {
    /* Ensure that this is the main thread. */
    if (pthread_self() != control_thread)
        stop_all();
}


#ifdef EMPTY_CYCLES
int num_bubble_cycles = 100;
#endif

int main (int argc, char *argv[])
{
    int rc;
    mic_num_lcores = sysconf(_SC_NPROCESSORS_ONLN);
    mic_num_pcores = sysconf(_SC_NPROCESSORS_ONLN) / KNAPP_MAX_THREADS_PER_CORE;
    memzero(pcore_used, KNAPP_NUM_CORES);
    log_info("%ld lcores (%ld pcores) detected.\n", mic_num_lcores, mic_num_pcores);

    setlocale(LC_NUMERIC, "");
#ifdef EMPTY_CYCLES
    if (argc > 1) {
        num_bubble_cycles = atoi(argv[1]);
        assert(num_bubble_cycles > 0);
        fprintf(stderr, "# of bubbles in kernel set to %d\n", num_bubble_cycles);
    } else {
        fprintf(stderr, "Need extra parameter for # of empty cycles\n");
        exit(EXIT_FAILURE);
    }
#endif
    exit_flag = false;
    pthread_attr_t attr;
    {
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
        size_t cpuset_sz = CPU_ALLOC_SIZE(mic_num_lcores);
        cpu_set_t *cpuset_master = CPU_ALLOC(mic_num_lcores);
        CPU_ZERO_S(cpuset_sz, cpuset_master);
        CPU_SET_S(0, cpuset_sz, cpuset_master); // pin to the first core.
        pthread_attr_setaffinity_np(&attr, cpuset_sz, cpuset_master);
        CPU_FREE(cpuset_master);
    }
    rc = pthread_create(&control_thread, &attr, control_thread_loop, nullptr);
    assert(0 == rc);
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);
    pthread_join(control_thread, nullptr);
    return 0;
}

// vim: ts=8 sts=4 sw=4 et
