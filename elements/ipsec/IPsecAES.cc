#include "IPsecAES.hh"
#ifdef USE_CUDA
#include "IPsecAES_kernel.hh"
#endif
#include "../../lib/types.hh"

using namespace std;
using namespace nba;

/* Array which stores per-tunnel AES key for each tunnel.
 * It is copied to each node's node local storage during per-node initialization
 * and freed in per-thread initialization.*/
struct aes_sa_entry *aes_sa_entry_array;
/* Map which stores (src-dst pair, tunnel index).
 * It is copied to each node's node local storage during per-node initialization*/
unordered_map<struct ipaddr_pair, int> aes_sa_table;

int IPsecAES::initialize()
{
    // Get ptr for CPU & GPU from the node-local storage.

    /* Storage for host ipsec tunnel index table */
    h_sa_table = (unordered_map<struct ipaddr_pair, int> *)ctx->node_local_storage->get_alloc("h_aes_sa_table");

    /* Storage for host aes key array */
    h_key_array = (struct aes_sa_entry *) ctx->node_local_storage->get_alloc("h_aes_key_array");

    /* Get device pointer from the node local storage. */
    d_key_array_ptr = ((memory_t *) ctx->node_local_storage->get_alloc("d_aes_key_array_ptr"))[0];

    if (aes_sa_entry_array != NULL) {
        free(aes_sa_entry_array);
        aes_sa_entry_array = NULL;
    }

    return 0;
}

int IPsecAES::initialize_global()
{
    // generate global table and array only once per element class.
    struct ipaddr_pair pair;
    struct aes_sa_entry *entry;
    unsigned char fake_iv[AES_BLOCK_SIZE] = {0};

    assert(num_tunnels != 0);
    aes_sa_entry_array = (struct aes_sa_entry *) malloc (sizeof(struct aes_sa_entry) *num_tunnels);
    for (int i = 0; i < num_tunnels; i++) {
        pair.src_addr  = 0x0a000001u;
        pair.dest_addr = 0x0a000000u | (i + 1); // (rand() % 0xffffff);
        auto result = aes_sa_table.insert(make_pair<ipaddr_pair&, int&>(pair, i));
        assert(result.second == true);

        entry = &aes_sa_entry_array[i];
        entry->entry_idx = i;
        rte_memcpy(entry->aes_key, "1234123412341234", AES_BLOCK_SIZE);
#ifdef USE_OPENSSL_EVP
    // TODO: check if copying globally initialized evpctx works okay.
    EVP_CIPHER_CTX_init(&entry->evpctx);
        //if (EVP_EncryptInit(&entry->evpctx, EVP_aes_128_ctr(), entry->aes_key, esph->esp_iv) != 1)
        if (EVP_EncryptInit(&entry->evpctx, EVP_aes_128_ctr(), entry->aes_key, fake_iv) != 1)
            fprintf(stderr, "IPsecAES: EVP_EncryptInit() - %s\n", ERR_error_string(ERR_get_error(), NULL));
#endif
        AES_set_encrypt_key((uint8_t *) entry->aes_key, 128, &entry->aes_key_t);
    }

    ERR_load_crypto_strings();

    return 0;
};

int IPsecAES::initialize_per_node()
{
    unordered_map<struct ipaddr_pair, int> *temp_table = NULL;
    struct aes_sa_entry *temp_array = NULL;
    struct ipaddr_pair key;
    int value, size;

    /* Storage for host ipsec tunnel index table */
    size = sizeof(unordered_map<struct ipaddr_pair, int>);
    ctx->node_local_storage->alloc("h_aes_sa_table", size);
    temp_table = (unordered_map<struct ipaddr_pair, int> *)ctx->node_local_storage->get_alloc("h_aes_sa_table");
    new (temp_table) unordered_map<struct ipaddr_pair, int>();

    for (auto iter = aes_sa_table.begin(); iter != aes_sa_table.end(); iter++) {
        key = iter->first;
        value = iter->second;
        temp_table->insert(make_pair<ipaddr_pair&, int&>(key, value));
    }

    /* Storage for host aes key array */
    size = sizeof(struct aes_sa_entry) * num_tunnels;
    ctx->node_local_storage->alloc("h_aes_key_array", size);
    temp_array = (struct aes_sa_entry *) ctx->node_local_storage->get_alloc("h_aes_key_array");
    assert(aes_sa_entry_array != NULL);
    rte_memcpy(temp_array, aes_sa_entry_array, size);

    /* Storage for pointer, which points aes key array in device */
    ctx->node_local_storage->alloc("d_aes_key_array_ptr", sizeof(memory_t));

    return 0;
}

int IPsecAES::configure(comp_thread_context *ctx, std::vector<std::string> &args)
{
    Element::configure(ctx, args);
    num_tunnels = 1024;            // TODO: this value must be come from configuration.
    return 0;
}

// Input packet: assumes ESP encaped, but payload not encrypted yet.
// +----------+---------------+--------+----+------------+---------+-------+---------------------+
// | Ethernet | IP(proto=ESP) |  ESP   | IP |  payload   | padding | extra | HMAC-SHA1 signature |
// +----------+---------------+--------+----+------------+---------+-------+---------------------+
// ^ethh      ^iph            ^esph    ^encrypt_ptr
//                                     <===== to be encrypted with AES ====>
//
int IPsecAES::process(int input_port, struct rte_mbuf *pkt, struct annotation_set *anno)
{
    struct ether_hdr *ethh = rte_pktmbuf_mtod(pkt, struct ether_hdr *);
    struct iphdr *iph = (struct iphdr *) (ethh + 1);
    struct esphdr *esph = (struct esphdr *) (iph + 1);
    uint8_t ecount_buf[AES_BLOCK_SIZE] = { 0 };

    // TODO: support decrpytion.
    uint8_t *encrypt_ptr = (uint8_t*) esph + sizeof(*esph);
    int encrypted_len = ntohs(iph->tot_len) - sizeof(struct iphdr) - sizeof(struct esphdr) - SHA_DIGEST_LENGTH;
    int pad_len = AES_BLOCK_SIZE - (encrypted_len + 2) % AES_BLOCK_SIZE;
    int enc_size = encrypted_len + pad_len + 2;    // additional two bytes mean the "extra" part.
    int err = 0;
    struct aes_sa_entry *sa_entry = NULL;

    if (likely(anno_isset(anno, NBA_ANNO_IPSEC_FLOW_ID))) {
        sa_entry = &h_key_array[anno_get(anno, NBA_ANNO_IPSEC_FLOW_ID)];
        unsigned mode = 0;
#ifdef USE_OPENSSL_EVP
        int cipher_body_len = 0;
        int cipher_add_len = 0;
        memcpy(sa_entry->evpctx.iv, esph->esp_iv, AES_BLOCK_SIZE);
        if (EVP_EncryptUpdate(&sa_entry->evpctx, encrypt_ptr, &cipher_body_len, encrypt_ptr, encrypted_len) != 1)
            fprintf(stderr, "IPsecAES: EVP_EncryptUpdate() - %s\n", ERR_error_string(ERR_get_error(), NULL));
        if (EVP_EncryptFinal(&sa_entry->evpctx, encrypt_ptr + cipher_body_len, &cipher_add_len) != 1)
            fprintf(stderr, "IPsecAES: EVP_EncryptFinal() - %s\n", ERR_error_string(ERR_get_error(), NULL));
#else
        AES_ctr128_encrypt(encrypt_ptr, encrypt_ptr, enc_size, &sa_entry->aes_key_t, esph->esp_iv, ecount_buf, &mode);
#endif
    } else {
    return DROP;
    }

    return 0;
}

#ifdef USE_CUDA
void IPsecAES::cuda_init_handler(ComputeDevice *device)
{
    // Put key array content to device space.
    long key_array_size = sizeof(struct aes_sa_entry) * num_tunnels;
    h_key_array = (struct aes_sa_entry *) ctx->node_local_storage->get_alloc("h_aes_key_array");
    memory_t key_array_in_device = device->alloc_device_buffer(key_array_size, 0);
    device->memwrite(h_key_array, key_array_in_device, 0, key_array_size);

    // Store the device pointer for per-thread instances.
    memory_t *p = (memory_t *) ctx->node_local_storage->get_alloc("d_aes_key_array_ptr");
    ((memory_t *) p)[0] = key_array_in_device;
}
#endif

void IPsecAES::preproc(int input_port, void *custom_input, struct rte_mbuf *pkt, struct annotation_set *anno)
{
    return;
}

void IPsecAES::prepare_input(ComputeContext *cctx, struct resource_param *res, struct annotation_set **anno_ptr_array)
{
    uint8_t *h_iv = nullptr;
    uint32_t *h_key_indice = nullptr;
    int32_t *h_pkt_offset = nullptr;
    uint16_t *h_pkt_index = nullptr;
    int32_t *h_block_offset = nullptr;
    memory_t d_iv;
    memory_t d_key_indice;
    memory_t d_pkt_offset;
    memory_t d_pkt_index;
    memory_t d_block_offset;

    /* The unit of computation in AES enc/decryption is "block",
     * which is composed of 16 byte slices of the packet payloads.
     * We rewrite the framework-provided resource parameters to
     * accommodate per-block parallelization. */
    size_t total_num_pkts = cctx->total_num_pkts;
    size_t num_blocks = (cctx->total_size_pkts + AES_BLOCK_SIZE - 1)
                        / AES_BLOCK_SIZE;

    res->num_workitems = num_blocks;
    res->num_threads_per_workgroup = 256;    // CUDA_THREADS_PER_AES_BLK;
    res->num_workgroups = (res->num_workitems + res->num_threads_per_workgroup - 1)
                          / res->num_threads_per_workgroup;

    size_t key_buffer_size          = AES_BLOCK_SIZE   * total_num_pkts;
    size_t offset_buffer_size       = sizeof(int32_t)  * total_num_pkts;
    size_t idx_buffer_size          = sizeof(uint16_t) * num_blocks;
    size_t block_offset_buffer_size = sizeof(int32_t)  * num_blocks;

    cctx->alloc_input_buffer(key_buffer_size, (void **) &h_iv, &d_iv);
    cctx->alloc_input_buffer(offset_buffer_size, (void **) &h_key_indice, &d_key_indice);

    cctx->alloc_input_buffer(offset_buffer_size, (void **) &h_pkt_offset, &d_pkt_offset);
    cctx->alloc_input_buffer(idx_buffer_size, (void **) &h_pkt_index, &d_pkt_index);
    cctx->alloc_input_buffer(block_offset_buffer_size, (void **) &h_block_offset, &d_block_offset);

    // variables for flow handling.
    struct aes_sa_entry *entry = NULL;
    convert_8B_to_1B_arr iv_first_half, iv_second_half;
    int idx = 0;
    int iv_offset = 0;

    unsigned global_block_cnt = 0;
    unsigned input_buffer_offset = 0;
    size_t *input_buffer_elemsizes = cctx->in_elemsizes_h;
    /* Per-packet loop for the task. */
    for (unsigned i = 0; i < total_num_pkts; ++i) {
        if (anno_ptr_array[i] == NULL) {
            h_pkt_offset[i] = -1;
            h_key_indice[i] = -1;
            // h_pkt_index and h_block_offset are per-block.
            // We just skip to set them here.
            continue;
        }
        uint64_t flow_id = anno_get(anno_ptr_array[i], NBA_ANNO_IPSEC_FLOW_ID);
        h_pkt_offset[i] = input_buffer_offset;
        h_key_indice[i] = (uint32_t) flow_id;
        // We store IVs in annotations since we CANNOT access the original
        // packets but only the annotations here.
        // TODO: could we improve the abstraction?
        iv_first_half.var = anno_get(anno_ptr_array[i], NBA_ANNO_IPSEC_IV1);
        iv_second_half.var = anno_get(anno_ptr_array[i], NBA_ANNO_IPSEC_IV2);
        for (idx = 0; idx < 8; idx++) {
            h_iv[iv_offset + idx] = iv_first_half.arr[idx];
            h_iv[iv_offset + idx + 8] = iv_second_half.arr[idx];
        }
        iv_offset += AES_BLOCK_SIZE;

        /* Per-block loop for the packet. */
        unsigned pkt_local_num_blocks = \
                (input_buffer_elemsizes[i] + AES_BLOCK_SIZE - 1) / AES_BLOCK_SIZE;
        for (unsigned j = 0; j < pkt_local_num_blocks; ++j) {
            unsigned global_block_idx = global_block_cnt + j;
            assert(global_block_idx < num_blocks);
            h_pkt_index[global_block_idx] = i;
            h_block_offset[global_block_idx] = j;
        }
        global_block_cnt    += pkt_local_num_blocks;
        input_buffer_offset += input_buffer_elemsizes[i];
    }
    assert(global_block_cnt <= num_blocks);

    cctx->host_ptr_storage[idx_iv] = h_iv;
    cctx->host_ptr_storage[idx_key_indice] = h_key_indice;
    cctx->host_ptr_storage[idx_pkt_offset] = h_pkt_offset;
    cctx->host_ptr_storage[idx_pkt_index] = h_pkt_index;
    cctx->host_ptr_storage[idx_block_offset] = h_block_offset;
    cctx->dev_mem_storage[idx_iv] = d_iv;
    cctx->dev_mem_storage[idx_key_indice] = d_key_indice;
    cctx->dev_mem_storage[idx_pkt_offset] = d_pkt_offset;
    cctx->dev_mem_storage[idx_pkt_index] = d_pkt_index;
    cctx->dev_mem_storage[idx_block_offset] = d_block_offset;
}

#ifdef USE_CUDA
void IPsecAES::cuda_compute_handler(ComputeContext *cctx, struct resource_param *res, struct annotation_set **anno_ptr_array)
{
    struct kernel_arg args[6];
    args[0].ptr = (void *) &cctx->dev_mem_storage[idx_iv].ptr;
    args[0].size = sizeof(void *);
    args[0].align = alignof(void *);
    args[1].ptr = (void *) &cctx->dev_mem_storage[idx_key_indice].ptr;
    args[1].size = sizeof(void *);
    args[1].align = alignof(void *);
    args[2].ptr = (void *) &d_key_array_ptr.ptr;
    args[2].size = sizeof(void *);
    args[2].align = alignof(void *);
    args[3].ptr = (void *) &cctx->dev_mem_storage[idx_pkt_offset].ptr;
    args[3].size = sizeof(void *);
    args[3].align = alignof(void *);
    args[4].ptr = (void *) &cctx->dev_mem_storage[idx_pkt_index].ptr;
    args[4].size = sizeof(void *);
    args[4].align = alignof(void *);
    args[5].ptr = (void *) &cctx->dev_mem_storage[idx_block_offset].ptr;
    args[5].size = sizeof(void *);
    args[5].align = alignof(void *);

    kernel_t kern;
    kern.ptr = ipsec_aes_encryption_get_cuda_kernel();
    cctx->enqueue_kernel_launch(kern, res, args, 6);
}
#endif

int IPsecAES::postproc(int input_port, void *custom_output, struct rte_mbuf *pkt, struct annotation_set *anno)
{
    /* ROI type of IPsecAES output packet: PACKET */
    // TODO: optional validation
    return 0;
}

size_t IPsecAES::get_desired_workgroup_size(const char *device_name) const
{
    #ifdef USE_CUDA
    if (!strcmp(device_name, "cuda"))
        return 256u;
    #endif
    #ifdef USE_PHI
    if (!strcmp(device_name, "phi"))
        return 256u;
    #endif
    return 256u;
}

// vim: ts=8 sts=4 sw=4 et
