#include "LookupIP6Route.hh"
#ifdef USE_CUDA
#include "LookupIP6Route_kernel.hh"
#endif
#include <nba/element/annotation.hh>
#include <nba/element/nodelocalstorage.hh>
#include <nba/framework/threadcontext.hh>
#include <nba/framework/computedevice.hh>
#include <nba/framework/computecontext.hh>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cerrno>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/ip6.h>
#include <rte_ether.h>
#include "util_routing_v6.hh"

#ifdef USE_CUDA
#include <cuda.h>
#include <nba/engines/cuda/utils.hh>
#endif
#ifdef USE_KNAPP
#include <nba/engines/knapp/kernels.hh>
#endif

using namespace std;
using namespace nba;

static uint64_t ntohll(uint64_t val)
{
        return ( (((val) >> 56) & 0x00000000000000ff) | (((val) >> 40) & 0x000000000000ff00) | \
                (((val) >> 24) & 0x0000000000ff0000) | (((val) >>  8) & 0x00000000ff000000) | \
                (((val) <<  8) & 0x000000ff00000000) | (((val) << 24) & 0x0000ff0000000000) | \
                (((val) << 40) & 0x00ff000000000000) | (((val) << 56) & 0xff00000000000000) );
}

LookupIP6Route::LookupIP6Route(): OffloadableElement()
{
    #ifdef USE_CUDA
    auto ch = [this](ComputeDevice *cdev, ComputeContext *ctx, struct resource_param *res) {
        this->accel_compute_handler(cdev, ctx, res);
    };
    offload_compute_handlers.insert({{"cuda", ch},});
    auto ih = [this](ComputeDevice *dev) { this->accel_init_handler(dev); };
    offload_init_handlers.insert({{"cuda", ih},});
    #endif
    #ifdef USE_KNAPP
    auto ch = [this](ComputeDevice *cdev, ComputeContext *ctx, struct resource_param *res) {
        this->accel_compute_handler(cdev, ctx, res);
    };
    offload_compute_handlers.insert({{"knapp.phi", ch},});
    auto ih = [this](ComputeDevice *dev) { this->accel_init_handler(dev); };
    offload_init_handlers.insert({{"knapp.phi", ih},});
    #endif

    num_tx_ports = 0;
    rr_port = 0;

    _table_ptr = NULL;
    _rwlock_ptr = NULL;
    d_tables = NULL;
    d_table_sizes = NULL;
}

int LookupIP6Route::configure(comp_thread_context *ctx, std::vector<std::string> &args)
{
    Element::configure(ctx, args);
    num_tx_ports = ctx->num_tx_ports;
    num_nodes = ctx->num_nodes;
    node_idx = ctx->loc.node_id;
    rr_port = 0;
    return 0;
}

int LookupIP6Route::initialize_global()
{
    // Generate table randomly..
    int seed = 7659243;
    int count = 200000;
    _original_table.from_random(seed, count);
    _original_table.build();
    return 0;
}

int LookupIP6Route::initialize_per_node()
{
    /* Storage for routing table. */
    ctx->node_local_storage->alloc("ipv6_table", sizeof(RoutingTableV6));
    RoutingTableV6 *table = (RoutingTableV6*)ctx->node_local_storage->get_alloc("ipv6_table");
    new (table) RoutingTableV6();

    // Copy table for the each node..
    _original_table.copy_to(table);

    /* Storage for device pointers. */
    ctx->node_local_storage->alloc("dev_tables", sizeof(Item*) * 128);
    ctx->node_local_storage->alloc("dev_table_sizes", sizeof(size_t) * 128);

    return 0;
}

int LookupIP6Route::initialize()
{
    // Called after coproc threads are initialized.

    /* Get routing table pointers from the node-local storage. */
    _table_ptr = (RoutingTableV6*)ctx->node_local_storage->get_alloc("ipv6_table");
    _rwlock_ptr = ctx->node_local_storage->get_rwlock("ipv6_table");

    /* Get GPU device pointers from the node-local storage. */
    d_tables      = (dev_mem_t *) ctx->node_local_storage->get_alloc("dev_tables");
    d_table_sizes = (dev_mem_t *) ctx->node_local_storage->get_alloc("dev_table_sizes");
    return 0;
}

/* The CPU version */
int LookupIP6Route::process(int input_port, Packet *pkt)
{
    struct ether_hdr *ethh = (struct ether_hdr *) pkt->data();
    struct ip6_hdr *ip6h   = (struct ip6_hdr *)(ethh + 1);
    uint128_t dest_addr;
    uint16_t lookup_result = 0xffff;
    memcpy(&dest_addr.u64[1], &ip6h->ip6_dst.s6_addr32[0], sizeof(uint64_t));
    memcpy(&dest_addr.u64[0], &ip6h->ip6_dst.s6_addr32[2], sizeof(uint64_t));
    dest_addr.u64[1] = ntohll(dest_addr.u64[1]);
    dest_addr.u64[0] = ntohll(dest_addr.u64[0]);

    // TODO: make an interface to set these locks to be
    // automatically handled by process_batch() method.
    //rte_rwlock_read_lock(_rwlock_ptr);
    lookup_result = _table_ptr->lookup((reinterpret_cast<uint128_t*>(&dest_addr)));
    //rte_rwlock_read_unlock(_rwlock_ptr);

    if (lookup_result == 0xffff) {
        /* Could not find destination. Use the second output for "error" packets. */
        pkt->kill();
        return 0;
    }

    #ifdef NBA_IPFWD_RR_NODE_LOCAL
    unsigned iface_in = anno_get(&pkt->anno, NBA_ANNO_IFACE_IN);
    unsigned n = (iface_in <= ((unsigned) num_tx_ports / 2) - 1) ? 0 : (num_tx_ports / 2);
    rr_port = (rr_port + 1) % (num_tx_ports / 2) + n;
    #else
    rr_port = (rr_port + 1) % (num_tx_ports);
    #endif
    anno_set(&pkt->anno, NBA_ANNO_IFACE_OUT, rr_port);
    output(0).push(pkt);
    return 0;
}

int LookupIP6Route::postproc(int input_port, void *custom_output, Packet *pkt)
{
    uint16_t lookup_result = *((uint16_t *)custom_output);
    if (lookup_result == 0xffff) {
        /* Could not find destination. Use the second output for "error" packets. */
        pkt->kill();
        return 0;
    }
    #ifdef NBA_IPFWD_RR_NODE_LOCAL
    unsigned iface_in = anno_get(&pkt->anno, NBA_ANNO_IFACE_IN);
    unsigned n = (iface_in <= ((unsigned) num_tx_ports / 2) - 1) ? 0 : (num_tx_ports / 2);
    rr_port = (rr_port + 1) % (num_tx_ports / 2) + n;
    #else
    rr_port = (rr_port + 1) % (num_tx_ports);
    #endif
    anno_set(&pkt->anno, NBA_ANNO_IFACE_OUT, rr_port);
    output(0).push(pkt);
    return 0;
}

size_t LookupIP6Route::get_desired_workgroup_size(const char *device_name) const
{
    #ifdef USE_CUDA
    if (!strcmp(device_name, "cuda"))
        return 256u;
    #endif
    #ifdef USE_PHI
    if (!strcmp(device_name, "phi"))
        return 256u; // TODO: confirm
    #endif
    return 256u;
}

void LookupIP6Route::accel_compute_handler(ComputeDevice *cdev,
                                           ComputeContext *cctx,
                                           struct resource_param *res)
{
    struct kernel_arg arg;
    void *ptr_args[2];
    ptr_args[0] = cdev->unwrap_device_buffer(*d_tables);
    arg = {&ptr_args[0], sizeof(void *), alignof(void *)};
    cctx->push_kernel_arg(arg);
    ptr_args[1] = cdev->unwrap_device_buffer(*d_table_sizes);
    arg = {&ptr_args[1], sizeof(void *), alignof(void *)};
    cctx->push_kernel_arg(arg);
    dev_kernel_t kern;
#ifdef USE_CUDA
    kern.ptr = ipv6_route_lookup_get_cuda_kernel();
#endif
#ifdef USE_KNAPP
    kern.ptr = (void *) (uintptr_t) knapp::ID_KERNEL_IPV6LOOKUP;
#endif
    cctx->enqueue_kernel_launch(kern, res);
}

void LookupIP6Route::accel_init_handler(ComputeDevice *device)
{
    size_t *table_sizes;
    void **table_ptrs;
    host_mem_t table_ptrs_h;  // <-> d_tables
    host_mem_t table_sizes_h; // <-> d_table_sizes

    /* Store the device pointers for per-thread instances. */
    d_tables      = (dev_mem_t *) ctx->node_local_storage->get_alloc("dev_tables");
    d_table_sizes = (dev_mem_t *) ctx->node_local_storage->get_alloc("dev_table_sizes");
    table_ptrs_h   = device->alloc_host_buffer(sizeof(void *) * 128, 0);
    table_sizes_h  = device->alloc_host_buffer(sizeof(size_t) * 128, 0);
    *d_tables      = device->alloc_device_buffer(sizeof(void *) * 128, 0, table_ptrs_h);
    *d_table_sizes = device->alloc_device_buffer(sizeof(size_t) * 128, 0, table_sizes_h);

    table_sizes = (size_t*) device->unwrap_host_buffer(table_sizes_h);
    table_ptrs  = (void **) device->unwrap_host_buffer(table_ptrs_h);

    /* table_ptrs_h keeps track of the temporary host-side references to tables in
     * the device for initialization and copy.
     * d_tables is the actual device buffer to store pointers in table_ptrs_h. */
    for (int i = 0; i < 128; i++) {
        table_sizes[i] = _original_table.m_Tables[i]->m_TableSize;
        size_t copy_size = sizeof(Item) * table_sizes[i] * 2;
        host_mem_t table_content_h;
        dev_mem_t table_content_d;
        fprintf(stderr, "ipv6 table alloc [%d]\n", i);
        table_content_h = device->alloc_host_buffer(copy_size, 0);
        table_content_d = device->alloc_device_buffer(copy_size, 0, table_content_h);
        table_ptrs[i] = device->unwrap_device_buffer(table_content_d);
        memcpy(device->unwrap_host_buffer(table_content_h), _original_table.m_Tables[i]->m_Table, copy_size);
        device->memwrite(table_content_h, table_content_d, 0, copy_size);
    }
    device->memwrite(table_ptrs_h, *d_tables, 0, sizeof(void *) * 128);
    device->memwrite(table_sizes_h, *d_table_sizes, 0, sizeof(size_t) * 128);

}

// vim: ts=8 sts=4 sw=4 et
