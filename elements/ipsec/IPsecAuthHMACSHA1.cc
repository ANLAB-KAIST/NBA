#include "IPsecAuthHMACSHA1.hh"
#ifdef USE_CUDA
#include "IPsecAuthHMACSHA1_kernel.hh"
#endif
#include "../../lib/types.hh"

using namespace std;
using namespace nba;

/* Array which stores per-tunnel HMAC key for each tunnel.
 * It is copied to each node's node local storage during per-node initialization
 * and freed in per-thread initialization.*/
struct hmac_sa_entry *hmac_sa_entry_array;
/* Map which stores (src-dst pair, tunnel index).
 * It is copied to each node's node local storage during per-node initialization*/
unordered_map<struct ipaddr_pair, int> hmac_sa_table;

int IPsecAuthHMACSHA1::initialize()
{
    // Get ptr for CPU & GPU pkt processing from the node-local storage.

    /* Storage for host ipsec tunnel index table */
    h_sa_table = (unordered_map<struct ipaddr_pair, int> *)ctx->node_local_storage->get_alloc("h_hmac_sa_table");

    /* Storage for host hmac key array */
    h_key_array = (struct hmac_sa_entry *) ctx->node_local_storage->get_alloc("h_hmac_key_array");

    /* Get device pointer from the node local storage. */
    d_key_array_ptr = ((struct hmac_sa_entry **)ctx->node_local_storage->get_alloc("d_hmac_key_array_ptr"))[0];

    if (hmac_sa_entry_array != NULL) {
        free(hmac_sa_entry_array);
        hmac_sa_entry_array = NULL;
    }

    return 0;
}

int IPsecAuthHMACSHA1::initialize_global()
{
    // generate global table and array only once per element class.
    struct ipaddr_pair pair;
    struct hmac_sa_entry *entry;

    assert(num_tunnels != 0);
    hmac_sa_entry_array = (struct hmac_sa_entry *) malloc(sizeof(struct hmac_sa_entry)*num_tunnels);

    for (int i = 0; i < num_tunnels; i++) {
        pair.src_addr  = 0x0a000001u;
        pair.dest_addr = 0x0a000000u | (i + 1); // (rand() % 0xffffff);
        auto result = hmac_sa_table.insert(make_pair<ipaddr_pair&, int&>(pair, i));
        assert(result.second == true);

        entry = &hmac_sa_entry_array[i];
        entry->entry_idx = i;
        rte_memcpy(&entry->hmac_key, "abcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcd", HMAC_KEY_SIZE);
    }

    return 0;
};

int IPsecAuthHMACSHA1::initialize_per_node()
{
    unordered_map<struct ipaddr_pair, int> *temp_table = NULL;
    struct hmac_sa_entry *temp_array = NULL;
    struct ipaddr_pair key;
    int value, size;

    /* Storage for host ipsec tunnel index table */
    size = sizeof(unordered_map<struct ipaddr_pair, int>);
    ctx->node_local_storage->alloc("h_hmac_sa_table", size);
    temp_table = (unordered_map<struct ipaddr_pair, int> *)ctx->node_local_storage->get_alloc("h_hmac_sa_table");
    new (temp_table) unordered_map<struct ipaddr_pair, int>();

    for (auto iter = hmac_sa_table.begin(); iter != hmac_sa_table.end(); iter++) {
        key = iter->first;
        value = iter->second;
        temp_table->insert(make_pair<ipaddr_pair&, int&>(key, value));
    }

    /* Storage for host hmac key array */
    size = sizeof(struct hmac_sa_entry) * num_tunnels;
    ctx->node_local_storage->alloc("h_hmac_key_array", size);
    temp_array = (struct hmac_sa_entry *) ctx->node_local_storage->get_alloc("h_hmac_key_array");
    assert(hmac_sa_entry_array != NULL);
    rte_memcpy(temp_array, hmac_sa_entry_array, size);

    /* Storage for pointer, which points hmac key array in device */
    ctx->node_local_storage->alloc("d_hmac_key_array_ptr", sizeof(void *));

    return 0;
}

int IPsecAuthHMACSHA1::configure(comp_thread_context *ctx, std::vector<std::string> &args)
{
    Element::configure(ctx, args);
    num_tunnels = 1024;         // TODO: this value must be come from configuration.

    return 0;
}

// Input packet: assumes encaped
// +----------+---------------+--------+----+------------+---------+-------+---------------------+
// | Ethernet | IP(proto=ESP) |  ESP   | IP |  payload   | padding | extra | HMAC-SHA1 signature |
// +----------+---------------+--------+----+------------+---------+-------+---------------------+
// ^ethh      ^iph            ^esph    ^encaped_iph
//                            ^payload_out
//                        ^encapsulated
//                            <===== authenticated part (payload_len) =====>
//
int IPsecAuthHMACSHA1::process(int input_port, struct rte_mbuf *pkt, struct annotation_set *anno)
{
    // We assume the size of hmac_key is less than 64 bytes.
    // TODO: check if input pkt is encapulated or not.
    struct ether_hdr *ethh = rte_pktmbuf_mtod(pkt, struct ether_hdr *);
    struct iphdr *iph = (struct iphdr *) (ethh + 1);
    struct esphdr *esph = (struct esphdr *) (iph + 1);
    uint8_t *encaped_iph = (uint8_t *) esph + sizeof(*esph);

    unsigned char *payload_out = (unsigned char*) ((uint8_t*)ethh + sizeof(struct ether_hdr)
                               + sizeof(struct iphdr));
    int payload_len = (ntohs(iph->tot_len) - (iph->ihl * 4) - SHA_DIGEST_LENGTH);
    uint8_t isum[SHA_DIGEST_LENGTH];
    uint8_t hmac_buf[2048];
    struct hmac_sa_entry *sa_entry;

    uint8_t *hmac_key;
    if (likely(anno_isset(anno, NBA_ANNO_IPSEC_FLOW_ID))) {
        sa_entry = &h_key_array[anno_get(anno, NBA_ANNO_IPSEC_FLOW_ID)];
        hmac_key = sa_entry->hmac_key;

        rte_memcpy(hmac_buf + 64, payload_out, payload_len);
        for (int i = 0; i < 8; i++)
            *((uint64_t*)hmac_buf + i) = 0x3636363636363636LLU ^ *((uint64_t*)hmac_key + i);
        SHA1(hmac_buf, 64 + payload_len, isum);

        rte_memcpy(hmac_buf + 64, isum, SHA_DIGEST_LENGTH);
        for (int i = 0; i < 8; i++) {
            *((uint64_t*)hmac_buf + i) = 0x5c5c5c5c5c5c5c5cLLU ^ *((uint64_t*)hmac_key + i);
        }
        SHA1(hmac_buf, 64 + SHA_DIGEST_LENGTH, payload_out + payload_len);
        // TODO: correctness check..

        /* HACK: preserve latency info from generator. */
        *(uint16_t*) ((char*) esph)                      = (uint16_t) anno_get(anno, NBA_ANNO_IFACE_OUT);
        *(uint64_t*) (((char*) esph) + sizeof(uint16_t)) =            anno_get(anno, NBA_ANNO_TIMESTAMP);
    } else {
        return DROP;
    }
    return 0;
}

#ifdef USE_CUDA
void IPsecAuthHMACSHA1::cuda_init_handler(ComputeDevice *device)
{
    // Put key array content to device space.
    long key_array_size = sizeof(struct hmac_sa_entry) * num_tunnels;
    h_key_array = (struct hmac_sa_entry *) ctx->node_local_storage->get_alloc("h_hmac_key_array");
    memory_t key_array_in_device = /*(struct hmac_sa_entry *)*/ device->alloc_device_buffer(key_array_size, 0);
    device->memwrite(h_key_array, key_array_in_device, 0, key_array_size);

    // Store the device pointer for per-thread instances.
    void *p = ctx->node_local_storage->get_alloc("d_hmac_key_array_ptr");
    ((memory_t *) p)[0] = key_array_in_device;
}

void IPsecAuthHMACSHA1::cuda_compute_handler(ComputeContext *cctx,
                                             struct resource_param *res,
                                             struct annotation_set **anno_ptr_array)
{
    struct kernel_arg args[3];
    args[0].ptr = (void *) &cctx->dev_mem_storage[idx_hmac_key_indice].ptr;
    args[0].size = sizeof(void *);
    args[0].align = alignof(void *);
    args[1].ptr = (void *) &d_key_array_ptr;
    args[1].size = sizeof(void *);
    args[1].align = alignof(void *);
    args[2].ptr = (void *) &cctx->dev_mem_storage[idx_pkt_offset].ptr;
    args[2].size = sizeof(void *);
    args[2].align = alignof(void *);

    kernel_t kern;
    kern.ptr = ipsec_hsha1_encryption_get_cuda_kernel();
    cctx->enqueue_kernel_launch(kern, res, args, 3);
}
#endif

void IPsecAuthHMACSHA1::preproc(int input_port, void *custom_input, struct rte_mbuf *pkt, struct annotation_set *anno)
{
    return;
}

void IPsecAuthHMACSHA1::prepare_input(ComputeContext *cctx,
                                      struct resource_param *res,
                                      struct annotation_set **anno_ptr_array)
{
    int *h_hmac_key_indice = nullptr;
    int32_t *h_pkt_offset = nullptr;
    memory_t d_hmac_key_indice;
    memory_t d_pkt_offset;

    // IPsec per-flow handling in GPU computation
    // Work unit of IPsecAuthHMACSHA1 is packet.
    size_t input_buffer_offset = 0;
    size_t *input_buffer_elemsizes = (size_t*) cctx->in_elemsizes_h;
    unsigned total_num_pkts = cctx->total_num_pkts;

    // initialize host, device buffer
    cctx->alloc_input_buffer(total_num_pkts * sizeof(int), (void **) &h_hmac_key_indice, &d_hmac_key_indice);
    cctx->alloc_input_buffer(total_num_pkts * sizeof(int32_t), (void **) &h_pkt_offset, &d_pkt_offset);

    struct annotation_set *pkt_anno_set = nullptr;
    struct hmac_sa_entry *entry = nullptr;

    // per-packet loop
    for (unsigned i = 0; i < total_num_pkts; ++i) {
        if (anno_ptr_array[i] == nullptr) {
            h_hmac_key_indice[i] = 0;
            h_pkt_offset[i] = -1;
            continue;
        }
        pkt_anno_set = anno_ptr_array[i];
        h_pkt_offset[i] = input_buffer_offset;
        input_buffer_offset += input_buffer_elemsizes[i];

        // Get tunnel index for a pkt and get HMAC key.
        h_hmac_key_indice[i] = (int) anno_get(pkt_anno_set, NBA_ANNO_IPSEC_FLOW_ID);
    }
    cctx->host_ptr_storage[idx_pkt_offset] = h_pkt_offset;
    cctx->host_ptr_storage[idx_hmac_key_indice] = h_hmac_key_indice;
    cctx->dev_mem_storage[idx_pkt_offset] = d_pkt_offset;
    cctx->dev_mem_storage[idx_hmac_key_indice] = d_hmac_key_indice;
}

int IPsecAuthHMACSHA1::postproc(int input_port, void *custom_output, struct rte_mbuf *pkt, struct annotation_set *anno)
{
    struct ether_hdr *ethh = rte_pktmbuf_mtod(pkt, struct ether_hdr *);
    struct iphdr *iph = (struct iphdr *) (ethh + 1);
    uint8_t *payload_out = (uint8_t *) (iph + 1);
    int payload_len = (ntohs(iph->tot_len) - (iph->ihl * 4) - SHA_DIGEST_LENGTH);

    // Copy result of HSHA1 is put into the HMAC-SHA1 signature part of input packet.
    rte_memcpy(payload_out + payload_len, custom_output, SHA_DIGEST_LENGTH);

    // TODO: optional validation

    /* HACK: preserve latency info from generator. */
    *(uint16_t*) ((char*) payload_out)                      = (uint16_t) anno_get(anno, NBA_ANNO_IFACE_OUT);
    *(uint64_t*) (((char*) payload_out) + sizeof(uint16_t)) =            anno_get(anno, NBA_ANNO_TIMESTAMP);

    // Just pass packet to the next element.
    return 0;
}

size_t IPsecAuthHMACSHA1::get_desired_workgroup_size(const char *device_name) const
{
    #ifdef USE_CUDA
    if (!strcmp(device_name, "cuda"))
        return 64u;
    #endif
    #ifdef USE_PHI
    if (!strcmp(device_name, "phi"))
        return 32u;
    #endif
    return 32u;
}
