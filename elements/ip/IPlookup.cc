#include "IPlookup.hh"
#ifdef USE_CUDA
#include "IPlookup_kernel.hh"
#endif
#include "../../lib/computecontext.hh"
#include "../../lib/types.hh"

using namespace std;
using namespace nba;

static unordered_map<uint32_t, uint16_t> pPrefixTable[33];
static unsigned int current_TBLlong = 0;

int IPlookup::initialize()
{
    /* Get routing table pointers from the node-local storage. */
    TBL24_h = (uint16_t *) ctx->node_local_storage->get_alloc("TBL24");
    p_rwlock_TBL24 = ctx->node_local_storage->get_rwlock("TBL24");
    TBLlong_h = (uint16_t *) ctx->node_local_storage->get_alloc("TBLlong");
    p_rwlock_TBLlong = ctx->node_local_storage->get_rwlock("TBLlong");

    /* Get device pointers from the node-local storage. */
    TBL24_d   = ((memory_t **) ctx->node_local_storage->get_alloc("TBL24_dev_ptr"))[0];
    TBLlong_d = ((memory_t **) ctx->node_local_storage->get_alloc("TBLlong_dev_ptr"))[0];
    return 0;
}

int IPlookup::initialize_global()
{
    // Loading IP forwarding table from file.
    // TODO: load it from parsed configuration.

    const char *filename = "configs/routing_info.txt";  // TODO: remove it or change it to configuration..
    printf("element::IPlookup: Loading the routing table entries from %s\n", filename);

    ipv4_load_rib_from_file(filename);
    current_TBLlong = 0;

    return 0;
}

int IPlookup::initialize_per_node()
{
    /* Storage for routing table. */
    ctx->node_local_storage->alloc("TBL24", sizeof(uint16_t) * ipv4_get_TBL24_size());
    ctx->node_local_storage->alloc("TBLlong", sizeof(uint16_t) * ipv4_get_TBLlong_size());
    /* Storage for device pointers. */
    ctx->node_local_storage->alloc("TBL24_dev_ptr", sizeof(memory_t));
    ctx->node_local_storage->alloc("TBLlong_dev_ptr", sizeof(memory_t));

    printf("element::IPlookup: Initializing FIB from the global RIB for NUMA node %d...\n", node_idx);
    ipv4_build_fib();

    return 0;
}

int IPlookup::configure(comp_thread_context *ctx, std::vector<std::string> &args)
{
    Element::configure(ctx, args);
    num_tx_ports = ctx->num_tx_ports;
    num_nodes = ctx->num_nodes;
    node_idx = ctx->loc.node_id;
    rr_port = 0;
    return 0;
}

/* The CPU version */
int IPlookup::process(int input_port, Packet *pkt)
{
    struct ether_hdr *ethh = (struct ether_hdr *) pkt->data();
    struct iphdr *iph   = (struct iphdr *)(ethh + 1);
    uint32_t dest_addr = ntohl(iph->daddr);
    uint16_t lookup_result = 0xffff;

    ipv4_route_lookup(dest_addr, &lookup_result);
    if (lookup_result == 0xffff) {
        /* Could not find destination. Use the second output for "error" packets. */
        pkt->kill();
        return 0;
    }

    //unsigned n = (pkt->pkt.in_port <= (num_tx_ports / 2) - 1) ? 0 : (num_tx_ports / 2);
    //rr_port = (rr_port + 1) % (num_tx_ports / 2) + n;
    rr_port = (rr_port + 1) % (num_tx_ports);
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

    //unsigned n = (pkt->pkt.in_port <= (num_tx_ports / 2) - 1) ? 0 : (num_tx_ports / 2);
    //rr_port = (rr_port + 1) % (num_tx_ports / 2) + n;
    rr_port = (rr_port + 1) % (num_tx_ports);
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
void IPlookup::cuda_compute_handler(ComputeContext *cctx,
                                    struct resource_param *res)
{
    //printf("G++ datablock_kernel_arg (%lu)\n", sizeof(struct datablock_kernel_arg));
    //printf("G++   .total_item_count (%lu)\n", offsetof(struct datablock_kernel_arg, total_item_count));
    //printf("G++   .buffer_bases (%lu)\n", offsetof(struct datablock_kernel_arg, buffer_bases));
    //printf("G++   .item_count (%lu)\n", offsetof(struct datablock_kernel_arg, item_count));
    //printf("G++   .item_size (%lu)\n", offsetof(struct datablock_kernel_arg, item_size));
    //printf("G++   .item_sizes (%lu)\n", offsetof(struct datablock_kernel_arg, item_sizes));

    struct kernel_arg arg;
    arg = {(void *) &TBL24_d, sizeof(void *), alignof(void *)};
    cctx->push_kernel_arg(arg);
    arg = {(void *) &TBLlong_d, sizeof(void *), alignof(void *)};
    cctx->push_kernel_arg(arg);
    kernel_t kern;
    kern.ptr = ipv4_route_lookup_get_cuda_kernel();
    cctx->enqueue_kernel_launch(kern, res);
}

void IPlookup::cuda_init_handler(ComputeDevice *device)
{
    memory_t new_TBL24_d   = /*(uint16_t *)*/ device->alloc_device_buffer(sizeof(uint16_t) * ipv4_get_TBL24_size(), HOST_TO_DEVICE);
    memory_t new_TBLlong_d = /*(uint16_t *)*/ device->alloc_device_buffer(sizeof(uint16_t) * ipv4_get_TBLlong_size(), HOST_TO_DEVICE);
    TBL24_h = (uint16_t *)   ctx->node_local_storage->get_alloc("TBL24");
    TBLlong_h = (uint16_t *) ctx->node_local_storage->get_alloc("TBLlong");
    device->memwrite(TBL24_h,   new_TBL24_d,   0, sizeof(uint16_t) * ipv4_get_TBL24_size());
    device->memwrite(TBLlong_h, new_TBLlong_d, 0, sizeof(uint16_t) * ipv4_get_TBLlong_size());

    /* Store the device pointers for per-thread instances. */
    memory_t *TBL24_dev_ptr_storage   = (memory_t *) ctx->node_local_storage->get_alloc("TBL24_dev_ptr");
    memory_t *TBLlong_dev_ptr_storage = (memory_t *) ctx->node_local_storage->get_alloc("TBLlong_dev_ptr");
    (TBL24_dev_ptr_storage)[0]   = new_TBL24_d;
    (TBLlong_dev_ptr_storage)[0] = new_TBLlong_d;
}
#endif

int IPlookup::ipv4_route_add(uint32_t addr, uint16_t len, uint16_t nexthop)
{
    pPrefixTable[len][addr] = nexthop;
    return 0;
}

int IPlookup::ipv4_route_del(uint32_t addr, uint16_t len)
{
    pPrefixTable[len].erase(addr);
    return 0;
}

int IPlookup::ipv4_build_fib()
{
    uint16_t *_TBL24    = (uint16_t *) ctx->node_local_storage->get_alloc("TBL24");
    uint16_t *_TBLlong  = (uint16_t *) ctx->node_local_storage->get_alloc("TBLlong");

    // ipv4_build_fib() is called for each node sequencially, before comp thread starts.
    // No rwlock protection is needed.
    memset(_TBL24, 0, TBL24_SIZE * sizeof(uint16_t));
    memset(_TBLlong, 0, TBLLONG_SIZE * sizeof(uint16_t));

    for (unsigned i = 0; i <= 24; i++) {
        for (auto it = pPrefixTable[i].begin(); it != pPrefixTable[i].end(); it++) {
            uint32_t addr = (*it).first;
            uint16_t dest = (uint16_t)(0xffffu & (uint64_t)(*it).second);
            uint32_t start = addr >> 8;
            uint32_t end = start + (0x1u << (24 - i));
            for (unsigned k = start; k < end; k++)
                _TBL24[k] = dest;
        }
    }

    for (unsigned i = 25; i <= 32; i++) {
        for (auto it = pPrefixTable[i].begin(); it != pPrefixTable[i].end(); it++) {
            uint32_t addr = (*it).first;
            uint16_t dest = (uint16_t)(0x0000ffff & (uint64_t)(*it).second);
            uint16_t dest24 = _TBL24[addr >> 8];
            if (((uint16_t)dest24 & 0x8000u) == 0) {
                uint32_t start = current_TBLlong + (addr & 0x000000ff);
                uint32_t end = start + (0x00000001u << (32 - i));

                for (unsigned j = current_TBLlong; j <= current_TBLlong + 256; j++)
                {
                    if (j < start || j >= end)
                        _TBLlong[j] = dest24;
                    else
                        _TBLlong[j] = dest;
                }
                _TBL24[addr >> 8]  = (uint16_t)(current_TBLlong >> 8) | 0x8000u;
                current_TBLlong += 256;
                assert(current_TBLlong <= TBLLONG_SIZE);
            } else {
                uint32_t start = ((uint32_t)dest24 & 0x7fffu) * 256 + (addr & 0x000000ff);
                uint32_t end = start + (0x00000001u << (32 - i));

                for (unsigned j = start; j < end; j++)
                    _TBLlong[j] = dest;
            }
        }
    }
    return 0;
}

// vim: ts=8 sts=4 sw=4 et
