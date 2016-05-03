#include <nba/engines/knapp/defs.hh>
#include <nba/engines/knapp/mictypes.hh>
#include <nba/engines/knapp/sharedtypes.hh>
#include <nba/engines/knapp/micbarrier.hh>
#include <nba/engines/knapp/micutils.hh>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <unistd.h>
#include <signal.h>
#include <locale.h>


extern int global_vdevice_counter;

extern std::map<nba::knapp::ctrl_msg_t, std::string> ctrltype_to_ctrlstring;
extern nba::knapp::worker_func_t worker_funcs[];

extern char **kernel_paths;
extern char **kernel_names;
std::vector<struct nba::knapp::vdevice *> vdevs;

using namespace nba;
using namespace nba::knapp;

cpu_set_t **cpuset_per_lcore;
pthread_attr_t *attr_per_lcore;
int core_util[KNAPP_NUM_CORES][KNAPP_MAX_THREADS_PER_CORE];

/* MIC daemon consists of 3 types of threads:
 * (1) control_thread_loop: global state mgmt (e.g., "cudaMalloc", "cudaMmecpy")
 * (2) master_thread_loop: vDevice controller
 * (3) worker_thread_loop: vDevice worker
 *
 * Each vDevice has at least one master and zero or more workers.
 */

namespace nba { namespace knapp {

void *control_thread_loop(void *arg);
void *master_thread_loop(void *arg);
void *worker_thread_loop(void *arg);
void stop_all();
void handle_signal(int signal);

}} // endns(nba::knapp)

static void *nba::knapp::worker_thread_loop(void *arg) {
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

static void *nba::knapp::master_thread_loop(void *arg) {
    struct vdevice *vdev = (struct vdevice *) arg;
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

    // REWRITE: Receive and set vdev workload type
    //recv_ctrlmsg(vdev->ctrl_epd, vdev->ctrlbuf, OP_SET_WORKLOAD_TYPE, &vdev->workload_type, &vdev->offload_batch_size, NULL, NULL);
    //log_device(vdev->device_id, "Workload type set to %s, offload batch size set to %u\n", proto_to_appstring[(knapp_proto_t) vdev->workload_type].c_str(), vdev->offload_batch_size);
    //send_ctrlresp(vdev->ctrl_epd, vdev->ctrlbuf, OP_SET_WORKLOAD_TYPE, NULL, NULL, NULL, NULL);

    // REWRITE: Receive malloc size, allocate, and return remote window
    //recv_ctrlmsg(vdev->ctrl_epd, vdev->ctrlbuf, OP_MALLOC, &vdev->inputbuf_size, &vdev->resultbuf_size, &vdev->pipeline_depth, &vdev->remote_writebuf_base_ra);
    //log_device(vdev->device_id, "writebuf_base: %ld\n", vdev->remote_writebuf_base_ra);
    //assert ( 0 == bufarray_ra_init(&vdev->inputbuf_array, vdev->pipeline_depth, vdev->inputbuf_size, PAGE_SIZE, vdev->data_epd, SCIF_PROT_READ | SCIF_PROT_WRITE) );
    //assert ( 0 == bufarray_ra_init(&vdev->resultbuf_array, vdev->pipeline_depth, vdev->resultbuf_size, PAGE_SIZE, vdev->data_epd, SCIF_PROT_READ | SCIF_PROT_WRITE) );
    //assert ( NULL != (vdev->data_ready_barriers = (Barrier **) _mm_malloc(sizeof(Barrier *) * vdev->pipeline_depth, CACHE_LINE_SIZE) ) );
    //assert ( NULL != (vdev->task_done_barriers = (Barrier **) _mm_malloc(sizeof(Barrier *) * vdev->pipeline_depth, CACHE_LINE_SIZE) ) );
    //for ( int i = 0; i < vdev->pipeline_depth; i++ ) {
    //    vdev->data_ready_barriers[i] = new Barrier(vdev->num_worker_threads, vdev->device_id, BARRIER_PROFILE_INTERVAL);
    //    vdev->task_done_barriers[i] = new Barrier(vdev->num_worker_threads, vdev->device_id, BARRIER_PROFILE_INTERVAL);
    //}
    //log_device(vdev->device_id, "Allocating %'d input and result buffers of size %'llu and %'llu each\n", vdev->pipeline_depth, vdev->inputbuf_size, vdev->resultbuf_size);
    //off_t inputbuf_base_ra = bufarray_get_ra(&vdev->inputbuf_array, 0);
    //send_ctrlresp(vdev->ctrl_epd, vdev->ctrlbuf, OP_MALLOC, &inputbuf_base_ra, NULL, NULL, NULL);

    // Receive base RA offset for remote poll ring, and return ras poll ring

    int32_t pollring_len;
    recv_ctrlmsg(vdev->ctrl_epd, vdev->ctrlbuf, OP_REG_POLLRING, &pollring_len, &vdev->remote_poll_ring_window, NULL, NULL);

    assert ( 0 == pollring_init(&vdev->poll_ring, pollring_len, vdev->data_epd) );
    uint64_t volatile *pollring = vdev->poll_ring.ring;

    for ( int i = 0; i < pollring_len; i++ ) {
        pollring[i] = KNAPP_COPY_PENDING;
    }

    log_device(vdev->device_id, "Allocating local poll ring of length %u. (Page-aligned to %d bytes) Remote poll ring window at %lld\n", vdev->poll_ring.len, vdev->poll_ring.alloc_bytes, vdev->remote_poll_ring_window);
    log_device(vdev->device_id, "Local poll ring registered at RA %lld\n", vdev->poll_ring.ring_ra);
    send_ctrlresp(vdev->ctrl_epd, vdev->ctrlbuf, OP_REG_POLLRING, &vdev->poll_ring.ring_ra, NULL, NULL, NULL);
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
    for ( unsigned i = 0, ithread = 0; i < vdev->cores.size(); i++ ) {
        for ( unsigned iht = 0; iht < vdev->ht_per_core; iht++ ) {
            int pcore = vdev->cores[i];
            int ht = get_least_utilized_ht(pcore);
            int lcore = mic_pcore_to_lcore(pcore, ht);
            //log_device(vdev->device_id, "Creating thread for lcore %d (%d, %d) and thread %d\n", lcore, pcore, ht, ithread);
            assert ( 0 == pthread_create(&vdev->worker_threads[ithread], &attr_per_lcore[lcore], worker_thread_loop, (void *) &vdev->thread_info_array[ithread]) );
            core_util[pcore][ht]++;
            ithread++;
        }
    }
    log_device(vdev->device_id, "Running processing daemon...\n");
    int32_t cur_task_id = 0;
    while ( true ) {
        // TODO: State safety check?
        while ( pollring[cur_task_id] != KNAPP_OFFLOAD_COMPLETE ) {
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
    struct vdevice *vdev = (struct vdevice *) arg;
    int backlog = 16;
    int rc = 0;
    log_device(vdev->device_id, "Listening on the master channel...\n");

    rc = scif_listen(vdev->master_epd, backlog);
    assert(0 == rc);
    rc = scif_accept(vdev->master_epd, &vdev->master_port,
                     &vdev->master_epd, SCIF_ACCEPT_SYNC);
    assert(0 == rc);

    // TODO: implement a control protocol loop.
    while (true) {
    }

    return nullptr;
}


static void nba::knapp::stop_all() {
    for (auto vdev : vdevs)
        vdev->exit = true;
    for (auto vdev : vdevs)
        pthread_join(vdev->master_thread, nullptr);
}

static void nba::knapp::handle_signal(int signal) {
    stop_all();
}


#ifdef EMPTY_CYCLES
int num_bubble_cycles = 100;
#endif

int main (int argc, char *argv[])
{
    //if (check_collision(PROGRAM_NAME, COLLISION_USE_TEMP | COLLISION_NOWAIT) < 0)
    //    return -1;
    long num_lcores = sysconf(_SC_NPROCESSORS_ONLN);
    long num_pcores = sysconf(_SC_NPROCESSORS_ONLN) / KNAPP_MAX_THREADS_PER_CORE;
    setlocale(LC_NUMERIC, "");
    log_info("%ld lcores (%ld pcores) detected.\n", num_lcores, num_pcores);

    cpuset_per_lcore = (cpu_set_t **)     _mm_malloc(sizeof(cpu_set_t *) * num_lcores, CACHE_LINE_SIZE);
    attr_per_lcore   = (pthread_attr_t *) _mm_malloc(sizeof(pthread_attr_t) * num_lcores, CACHE_LINE_SIZE);
    size_t cpu_setsize = CPU_ALLOC_SIZE(num_lcores);
    memset(core_util, 0, sizeof(core_util));

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
    for (int lcore = 0; lcore < num_lcores; lcore++) {
        int rc;
        cpuset_per_lcore[lcore] = CPU_ALLOC(num_lcores);
        if (cpuset_per_lcore[lcore] == NULL) {
            perror("CPU_ALLOC");
            exit(EXIT_FAILURE);
        }
        CPU_ZERO_S(cpu_setsize, cpuset_per_lcore[lcore]);
        CPU_SET_S(lcore, cpu_setsize, cpuset_per_lcore[lcore]); // ROTATE BY 1 (Knights Corner specific)
        pthread_attr_init(&attr_per_lcore[lcore]);
        assert(0 == pthread_attr_setdetachstate(&attr_per_lcore[lcore], PTHREAD_CREATE_JOINABLE));
        assert(0 == pthread_attr_setaffinity_np(&attr_per_lcore[lcore], cpu_setsize, cpuset_per_lcore[lcore]));
    }
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);
    for ( unsigned ivdev = 0; ivdev < vdevs.size(); ivdev++ ) {
        struct vdevice *vdev = vdevs[ivdev];
        int pcore_to_pin = vdev->cores[0];
        int ht_to_pin = get_least_utilized_ht(pcore_to_pin);
        int lcore = mic_pcore_to_lcore(pcore_to_pin, ht_to_pin);
        vdev->master_cpu = lcore;
        assert ( 0 == pthread_create(&vdev->master_thread, &attr_per_lcore[lcore], master_thread_loop, (void *) vdev) );
        core_util[pcore_to_pin][ht_to_pin]++;
    }
    for ( unsigned i = 0; i < vdevs.size(); i++ ) {
        pthread_join(vdevs[i]->master_thread, NULL);
    }
    return 0;
}

// vim: ts=8 sts=4 sw=4 et
