#include "IPsecAES.hh"
#ifdef USE_CUDA
#include "IPsecAES_kernel.hh"
#endif
#include <nba/element/annotation.hh>
#include <nba/element/nodelocalstorage.hh>
#include <nba/framework/threadcontext.hh>
#include <nba/framework/computedevice.hh>
#include <nba/framework/computecontext.hh>
#include <netinet/ip.h>
#include <openssl/evp.h>
#include <openssl/err.h>
#include <openssl/aes.h>
#include <openssl/sha.h>
#include "util_esp.hh"
#include "util_ipsec_key.hh"
#include "util_sa_entry.hh"
#include <rte_memory.h>
#include <rte_ether.h>

using namespace std;
using namespace nba;

/* Array which stores per-tunnel AES key for each tunnel.
 * It is copied to each node's node local storage during per-node initialization
 * and freed in per-thread initialization.*/
struct aes_sa_entry *aes_sa_entry_array;
/* Map which stores (src-dst pair, tunnel index).
 * It is copied to each node's node local storage during per-node initialization*/
unordered_map<struct ipaddr_pair, int> aes_sa_table;

IPsecAES::IPsecAES(): OffloadableElement()
{
    #ifdef USE_CUDA
    auto ch = [this](ComputeContext *ctx, struct resource_param *res) { this->cuda_compute_handler(ctx, res); };
    offload_compute_handlers.insert({{"cuda", ch},});
    auto ih = [this](ComputeDevice *dev) { this->cuda_init_handler(dev); };
    offload_init_handlers.insert({{"cuda", ih},});
    #endif
    num_tunnels = 0;
}

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
int IPsecAES::process(int input_port, Packet *pkt)
{
    struct ether_hdr *ethh = (struct ether_hdr *) pkt->data();
    struct iphdr *iph      = (struct iphdr *) (ethh + 1);
    struct esphdr *esph    = (struct esphdr *) (iph + 1);
    uint8_t ecount_buf[AES_BLOCK_SIZE] = { 0 };

    // TODO: support decrpytion.
    uint8_t *encrypt_ptr = (uint8_t*) esph + sizeof(*esph);
    int encrypted_len = ntohs(iph->tot_len) - sizeof(struct iphdr) - sizeof(struct esphdr) - SHA_DIGEST_LENGTH;
    int pad_len = AES_BLOCK_SIZE - (encrypted_len + 2) % AES_BLOCK_SIZE;
    int enc_size = encrypted_len + pad_len + 2;    // additional two bytes mean the "extra" part.
    int err = 0;
    struct aes_sa_entry *sa_entry = NULL;

    if (likely(anno_isset(&pkt->anno, NBA_ANNO_IPSEC_FLOW_ID))) {
        sa_entry = &h_key_array[anno_get(&pkt->anno, NBA_ANNO_IPSEC_FLOW_ID)];
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
        pkt->kill();
        return 0;
    }

    output(0).push(pkt);
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

#ifdef USE_CUDA
void IPsecAES::cuda_compute_handler(ComputeContext *cctx, struct resource_param *res)
{
    struct kernel_arg arg;
    arg = {(void *) &d_key_array_ptr.ptr, sizeof(void *), alignof(void *)};
    cctx->push_kernel_arg(arg);

    kernel_t kern;
    kern.ptr = ipsec_aes_encryption_get_cuda_kernel();
    cctx->enqueue_kernel_launch(kern, res);

    // TODO: data-copy-opt
    // ?->inc_dev_ver(dbid_ipsec_...);
}
#endif

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

int IPsecAES::postproc(int input_port, void *custom_output, Packet *pkt)
{
    output(0).push(pkt);
    return 0;
}

// vim: ts=8 sts=4 sw=4 et
