#include <nba/core/offloadtypes.hh>
#include <nba/framework/threadcontext.hh>
#include <nba/framework/computedevice.hh>
#include <nba/framework/computecontext.hh>
#include <nba/element/annotation.hh>
#include <nba/element/nodelocalstorage.hh>
#include <cstdio>
#include <cassert>
#include <arpa/inet.h>
#include <rte_ether.h>
#include <rte_ip.h>
#include "ip_route_core.hh"
#include "IPlookup.hh"
#ifdef USE_CUDA
#include "IPlookup_kernel.hh"
#endif

using namespace std;
using namespace nba;

IPlookup::IPlookup() : OffloadableElement(),
    num_tx_ports(0), rr_port(0),
    p_rwlock_TBL24(nullptr), p_rwlock_TBLlong(nullptr), tables(),
    TBL24_h(nullptr), TBLlong_h(nullptr), TBL24_d{nullptr}, TBLlong_d{nullptr}
{
    #ifdef USE_CUDA
    auto ch = [this](ComputeContext *ctx, struct resource_param *res) { this->cuda_compute_handler(ctx, res); };
    offload_compute_handlers.insert({{"cuda", ch},});
    auto ih = [this](ComputeDevice *dev) { this->cuda_init_handler(dev); };
    offload_init_handlers.insert({{"cuda", ih},});
    #endif

    num_tx_ports = 0;
    rr_port = 0;
    p_rwlock_TBL24 = nullptr;
    p_rwlock_TBLlong = nullptr;
    TBL24_h = nullptr;
    TBLlong_h = nullptr;
    TBL24_d = { nullptr };
    TBLlong_d = { nullptr };
}

int IPlookup::initialize_global()
{
    // Loading IP forwarding table from file.
    // TODO: load it from parsed configuration.

    const char *filename = "configs/routing_info.txt";  // TODO: remove it or change it to configuration..
    printf("element::IPlookup: Loading the routing table entries from %s\n", filename);

    ipv4route::load_rib_from_file(tables, filename);
    return 0;
}

int IPlookup::initialize_per_node()
{
    /* Storage for routing table. */
    ctx->node_local_storage->alloc("TBL24", sizeof(uint16_t) * ipv4route::get_TBL24_size());
    ctx->node_local_storage->alloc("TBLlong", sizeof(uint16_t) * ipv4route::get_TBLlong_size());
    /* Storage for device pointers. */
    ctx->node_local_storage->alloc("TBL24_dev_ptr", sizeof(dev_mem_t));
    ctx->node_local_storage->alloc("TBLlong_dev_ptr", sizeof(dev_mem_t));

    printf("element::IPlookup: Initializing FIB from the global RIB for NUMA node %d...\n", node_idx);
    ipv4route::build_direct_fib(tables,
        (uint16_t *) ctx->node_local_storage->get_alloc("TBL24"),
        (uint16_t *) ctx->node_local_storage->get_alloc("TBLlong"));

    return 0;
}

int IPlookup::initialize()
{
    /* Get routing table pointers from the node-local storage. */
    TBL24_h = (uint16_t *) ctx->node_local_storage->get_alloc("TBL24");
    TBLlong_h = (uint16_t *) ctx->node_local_storage->get_alloc("TBLlong");
    assert(TBL24_h != nullptr);
    assert(TBLlong_h != nullptr);
    p_rwlock_TBL24 = ctx->node_local_storage->get_rwlock("TBL24");
    p_rwlock_TBLlong = ctx->node_local_storage->get_rwlock("TBLlong");

    /* Get device pointers from the node-local storage. */
    TBL24_d   = (dev_mem_t *) ctx->node_local_storage->get_alloc("TBL24_dev_ptr");
    TBLlong_d = (dev_mem_t *) ctx->node_local_storage->get_alloc("TBLlong_dev_ptr");

    rr_port = 0;
    return 0;
}

int IPlookup::configure(comp_thread_context *ctx, std::vector<std::string> &args)
{
    Element::configure(ctx, args);
    num_tx_ports = ctx->num_tx_ports;
    num_nodes = ctx->num_nodes;
    node_idx = ctx->loc.node_id;
    return 0;
}

/* The CPU version */
int IPlookup::process(int input_port, Packet *pkt)
{
    struct ether_hdr *ethh = (struct ether_hdr *) pkt->data();
    struct ipv4_hdr *iph   = (struct ipv4_hdr *)(ethh + 1);
    uint32_t dest_addr = ntohl(iph->dst_addr);
    uint16_t lookup_result = 0xffff;

    ipv4route::direct_lookup(TBL24_h, TBLlong_h, dest_addr, &lookup_result);
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

int IPlookup::postproc(int input_port, void *custom_output, Packet *pkt)
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

size_t IPlookup::get_desired_workgroup_size(const char *device_name) const
{
    #ifdef USE_CUDA
    if (!strcmp(device_name, "cuda"))
        return 512u;
    #endif
    #ifdef USE_PHI
    if (!strcmp(device_name, "phi"))
        return 256u;
    #endif
    return 256u;
}

#ifdef USE_CUDA
void IPlookup::cuda_init_handler(ComputeDevice *device)
{
    /* Store the device pointers for per-thread element instances. */
    size_t TBL24_alloc_size   = sizeof(uint16_t) * ipv4route::get_TBL24_size();
    size_t TBLlong_alloc_size = sizeof(uint16_t) * ipv4route::get_TBLlong_size();
    // As it is before initialize() is called, we need to get the pointers
    // from the node-local storage by ourselves here.
    uint16_t *_TBL24_h = nullptr;
    uint16_t *_TBLlong_h = nullptr;
    _TBL24_h = (uint16_t *) ctx->node_local_storage->get_alloc("TBL24");
    _TBLlong_h = (uint16_t *) ctx->node_local_storage->get_alloc("TBLlong");
    TBL24_d   = (dev_mem_t *) ctx->node_local_storage->get_alloc("TBL24_dev_ptr");
    TBLlong_d = (dev_mem_t *) ctx->node_local_storage->get_alloc("TBLlong_dev_ptr");
    *TBL24_d   = device->alloc_device_buffer(TBL24_alloc_size);
    *TBLlong_d = device->alloc_device_buffer(TBLlong_alloc_size);
    /* Convert host-side routing table to host_mem_t and copy the routing table. */
    device->memwrite({(void *) _TBL24_h},   *TBL24_d,   0, TBL24_alloc_size);
    device->memwrite({(void *) _TBLlong_h}, *TBLlong_d, 0, TBLlong_alloc_size);
}

void IPlookup::cuda_compute_handler(ComputeContext *cctx,
                                    struct resource_param *res)
{
    struct kernel_arg arg;
    void *ptr_args[2];
    ptr_args[0] = cctx->unwrap_device_buffer(*TBL24_d);
    arg = {(void *) &ptr_args[0], sizeof(void *), alignof(void *)};
    cctx->push_kernel_arg(arg);
    ptr_args[1] = cctx->unwrap_device_buffer(*TBLlong_d);
    arg = {(void *) &ptr_args[1], sizeof(void *), alignof(void *)};
    cctx->push_kernel_arg(arg);
    dev_kernel_t kern;
    kern.ptr = ipv4_route_lookup_get_cuda_kernel();
    cctx->enqueue_kernel_launch(kern, res);
}
#endif

// vim: ts=8 sts=4 sw=4 et
