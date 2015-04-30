#include "IPsecHMACSHA1AES.hh"
#ifdef USE_CUDA
#include "IPsecHMACSHA1AES_kernel.hh"
#endif
#include "../../lib/types.hh"

using namespace std;
using namespace nba;

/* Array which stores per-tunnel HMAC & AES key for each tunnel.
 * It is copied to each node's node local storage during per-node initialization
 * and freed in per-thread initialization.*/
struct hmac_aes_sa_entry *hmac_aes_sa_entry_array;
/* Map which stores (src-dst pair, tunnel index).
 * It is copied to each node's node local storage during per-node initialization*/
unordered_map<struct ipaddr_pair, int> hmac_aes_sa_table;

int IPsecHMACSHA1AES::initialize()
{
    // Get ptr for CPU & GPU from the node-local storage.

    /* Storage for host ipsec tunnel index table */
    h_sa_table = (unordered_map<struct ipaddr_pair, int> *)ctx->node_local_storage->get_alloc("h_hmac_aes_sa_table");

    /* Storage for host hmac & aes key array */
    h_key_array = (struct hmac_aes_sa_entry *) ctx->node_local_storage->get_alloc("h_hmac_aes_key_array");

    /* Get device pointer from the node local storage. */
    d_key_array_ptr = ((memory_t *) ctx->node_local_storage->get_alloc("d_hmac_aes_key_array_ptr"))[0];

    if (hmac_aes_sa_entry_array != NULL) {
        free(hmac_aes_sa_entry_array);
        hmac_aes_sa_entry_array = NULL;
    }

    return 0;
}

int IPsecHMACSHA1AES::initialize_global()
{
    // generate global table and array only once per element class.
    struct ipaddr_pair pair;
    struct hmac_aes_sa_entry *entry;
    unsigned char fake_iv[AES_BLOCK_SIZE] = {0};

    assert(num_tunnels != 0);
    hmac_aes_sa_entry_array = (struct hmac_aes_sa_entry *) malloc (sizeof(struct hmac_aes_sa_entry) *num_tunnels);
    for (int i = 0; i < num_tunnels; i++) {
        pair.src_addr  = 0x0a000001u;
        pair.dest_addr = 0x0a000000u | (i + 1); // (rand() % 0xffffff);
        auto result = hmac_aes_sa_table.insert(make_pair<ipaddr_pair&, int&>(pair, i));
        assert(result.second == true);

        // HMAC key initialization
        rte_memcpy(&entry->hmac_key, "abcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcd", HMAC_KEY_SIZE);

        // AES key initialization
        entry = &hmac_aes_sa_entry_array[i];
        entry->entry_idx = i;
        rte_memcpy(entry->aes_key, "1234123412341234", AES_BLOCK_SIZE);
#ifdef USE_OPENSSL_EVP
    // TODO: check if copying globally initialized evpctx works okay.
    EVP_CIPHER_CTX_init(&entry->evpctx);
        //if (EVP_EncryptInit(&entry->evpctx, EVP_aes_128_cbc(), entry->aes_key, esph->esp_iv) != 1)
        if (EVP_EncryptInit(&entry->evpctx, EVP_aes_128_cbc(), entry->aes_key, fake_iv) != 1)
            fprintf(stderr, "IPsecHMACSHA1AES: EVP_EncryptInit() - %s\n", ERR_error_string(ERR_get_error(), NULL));
#endif
        AES_set_encrypt_key((uint8_t *) entry->aes_key, 128, &entry->aes_key_t);
    }

    ERR_load_crypto_strings();

    return 0;
};

int IPsecHMACSHA1AES::initialize_per_node()
{
    unordered_map<struct ipaddr_pair, int> *temp_table = NULL;
    struct hmac_aes_sa_entry *temp_array = NULL;
    struct ipaddr_pair key;
    int value, size;

    /* Storage for host ipsec tunnel index table */
    size = sizeof(unordered_map<struct ipaddr_pair, int>);
    ctx->node_local_storage->alloc("h_hmac_aes_sa_table", size);
    temp_table = (unordered_map<struct ipaddr_pair, int> *)ctx->node_local_storage->get_alloc("h_hmac_aes_sa_table");
    new (temp_table) unordered_map<struct ipaddr_pair, int>();

    for (auto iter = hmac_aes_sa_table.begin(); iter != hmac_aes_sa_table.end(); iter++) {
        key = iter->first;
        value = iter->second;
        temp_table->insert(make_pair<ipaddr_pair&, int&>(key, value));
    }

    /* Storage for host hmac & aes key array */
    size = sizeof(struct hmac_aes_sa_entry) * num_tunnels;
    ctx->node_local_storage->alloc("h_hmac_aes_key_array", size);
    temp_array = (struct hmac_aes_sa_entry *) ctx->node_local_storage->get_alloc("h_hmac_aes_key_array");
    assert(hmac_aes_sa_entry_array != NULL);
    rte_memcpy(temp_array, hmac_aes_sa_entry_array, size);

    /* Storage for pointer, which points hmac & aes key array in device */
    ctx->node_local_storage->alloc("d_hmac_aes_key_array_ptr", sizeof(memory_t));

    return 0;
}

int IPsecHMACSHA1AES::configure(comp_thread_context *ctx, std::vector<std::string> &args)
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
int IPsecHMACSHA1AES::process(int input_port, Packet *pkt)
{
    struct ether_hdr *ethh = rte_pktmbuf_mtod(pkt, struct ether_hdr *);
    struct iphdr *iph = (struct iphdr *) (ethh + 1);
    struct esphdr *esph = (struct esphdr *) (iph + 1);

    // HMAC-related variables
    unsigned char *payload_out = (unsigned char*) ((uint8_t*)ethh + sizeof(struct ether_hdr)
                                   + sizeof(struct iphdr));
    int payload_len = (ntohs(iph->tot_len) - (iph->ihl * 4) - SHA_DIGEST_LENGTH);
    uint8_t isum[SHA_DIGEST_LENGTH];
    uint8_t hmac_buf[2048];

    // AES-related variables
    uint8_t ecount_buf[AES_BLOCK_SIZE] = { 0 };
    uint8_t *encrypt_ptr = (uint8_t*) esph + sizeof(*esph);
    int encrypted_len = ntohs(iph->tot_len) - sizeof(struct iphdr) - sizeof(struct esphdr) - SHA_DIGEST_LENGTH;
    int pad_len = AES_BLOCK_SIZE - (encrypted_len + 2) % AES_BLOCK_SIZE;
    int enc_size = encrypted_len + pad_len + 2;    // additional two bytes mean the "extra" part.
    int err = 0;

    struct hmac_aes_sa_entry *sa_entry = NULL;
    uint8_t *hmac_key = NULL;

    if (likely(anno_isset(&pkt->anno, NBA_ANNO_IPSEC_FLOW_ID))) {
        sa_entry = &h_key_array[anno_get(&pkt->anno, NBA_ANNO_IPSEC_FLOW_ID)];

        // AES processing
        // TODO: support decrpytion.
        unsigned mode = 0;
#ifdef USE_OPENSSL_EVP
        int cipher_body_len = 0;
        int cipher_add_len = 0;
        memcpy(sa_entry->evpctx.iv, esph->esp_iv, AES_BLOCK_SIZE);
        if (EVP_EncryptUpdate(&sa_entry->evpctx, encrypt_ptr, &cipher_body_len, encrypt_ptr, encrypted_len) != 1)
            fprintf(stderr, "IPsecHMACSHA1AES: EVP_EncryptUpdate() - %s\n", ERR_error_string(ERR_get_error(), NULL));
        if (EVP_EncryptFinal(&sa_entry->evpctx, encrypt_ptr + cipher_body_len, &cipher_add_len) != 1)
            fprintf(stderr, "IPsecHMACSHA1AES: EVP_EncryptFinal() - %s\n", ERR_error_string(ERR_get_error(), NULL));
#else
        AES_cbc_encrypt(encrypt_ptr, encrypt_ptr, enc_size, &sa_entry->aes_key_t, esph->esp_iv, AES_ENCRYPT);
#endif

        // HMAC-SHA1 processing
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

    } else {
    return DROP;
    }

    return 0;
}

#ifdef USE_CUDA
void IPsecHMACSHA1AES::cuda_init_handler(ComputeDevice *device)
{
    // Put key array content to device space.
    long key_array_size = sizeof(struct hmac_aes_sa_entry) * num_tunnels;
    h_key_array = (struct hmac_aes_sa_entry *) ctx->node_local_storage->get_alloc("h_hmac_aes_key_array");
    memory_t key_array_in_device = device->alloc_device_buffer(key_array_size, 0);
    device->memwrite(h_key_array, key_array_in_device, 0, key_array_size);

    // Store the device pointer for per-thread instances.
    memory_t *p = (memory_t *) ctx->node_local_storage->get_alloc("d_hmac_aes_key_array_ptr");
    ((memory_t *) p)[0] = key_array_in_device;
}
#endif

void IPsecHMACSHA1AES::preproc(int input_port, void *custom_input, Packet *pkt)
{
    return;
}

void IPsecHMACSHA1AES::prepare_input(ComputeContext *cctx, struct resource_param *res, struct annotation_set **anno_ptr_array)
{
    uint8_t *h_iv = nullptr;
    uint32_t *h_key_indice = nullptr;
    int32_t *h_pkt_offset = nullptr;
    memory_t d_iv;
    memory_t d_key_indice;
    memory_t d_pkt_offset;

    /* The unit of computation in AES enc/decryption & HMAC-SHA1 hashing is "packet".
     * So AES encryption uses CBC mode. */

    size_t total_num_pkts = cctx->total_num_pkts;

    size_t key_buffer_size          = AES_BLOCK_SIZE   * total_num_pkts;
    size_t offset_buffer_size       = sizeof(int32_t)  * total_num_pkts;

    cctx->alloc_input_buffer(key_buffer_size, (void **) &h_iv, &d_iv);
    cctx->alloc_input_buffer(offset_buffer_size, (void **) &h_key_indice, &d_key_indice);
    cctx->alloc_input_buffer(offset_buffer_size, (void **) &h_pkt_offset, &d_pkt_offset);

    // variables for flow handling.
    struct hmac_aes_sa_entry *entry = NULL;
    convert_8B_to_1B_arr iv_first_half, iv_second_half;
    int idx = 0;
    int iv_offset = 0;

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
        input_buffer_offset += input_buffer_elemsizes[i];
    }

    cctx->host_ptr_storage[idx_iv] = h_iv;
    cctx->host_ptr_storage[idx_key_indice] = h_key_indice;
    cctx->host_ptr_storage[idx_pkt_offset] = h_pkt_offset;
    cctx->dev_mem_storage[idx_iv] = d_iv;
    cctx->dev_mem_storage[idx_key_indice] = d_key_indice;
    cctx->dev_mem_storage[idx_pkt_offset] = d_pkt_offset;
}

#ifdef USE_CUDA
void IPsecHMACSHA1AES::cuda_compute_handler(ComputeContext *cctx, struct resource_param *res, struct annotation_set **anno_ptr_array)
{
    struct kernel_arg args[4];
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

    kernel_t kern;
    kern.ptr = ipsec_hmac_sha1_aes_get_cuda_kernel();
    cctx->enqueue_kernel_launch(kern, res, args, 4);
}
#endif

int IPsecHMACSHA1AES::postproc(int input_port, void *custom_output, Packet *pkt)
{
    /* ROI type of IPsecHMACSHA1AES output packet: PACKET */
    // TODO: optional validation
    return 0;
}

size_t IPsecHMACSHA1AES::get_desired_workgroup_size(const char *device_name) const
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
