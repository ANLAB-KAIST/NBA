#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <unordered_map>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
#include <nba/framework/datablock.hh>
#include <nba/framework/datablock_shared.hh>
#include <nba/element/annotation.hh>
#include <nba/element/packet.hh>
#include <nba/element/packetbatch.hh>
#include <nba/framework/test_utils.hh>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <openssl/evp.h>
#include <openssl/err.h>
#include <openssl/aes.h>
#include <openssl/sha.h>
#include <rte_mbuf.h>
#include "../elements/ipsec/util_esp.hh"
#include "../elements/ipsec/util_ipsec_key.hh"
#include "../elements/ipsec/util_sa_entry.hh"
#ifdef USE_CUDA
#include "../elements/ipsec/IPsecAES_kernel.hh"
#include "../elements/ipsec/IPsecAuthHMACSHA1_kernel.hh"
#endif
#include "../elements/ipsec/IPsecDatablocks.hh"
/*
#require <lib/datablock.o>
#require <lib/test_utils.o>
#require "../elements/ipsec/IPsecDatablocks.o"
*/
#ifdef USE_CUDA
/*
#require "../elements/ipsec/IPsecAES_kernel.o"
#require "../elements/ipsec/IPsecAuthHMACSHA1_kernel.o"
*/
#endif

using namespace std;
using namespace nba;

#ifdef USE_CUDA

static int getNumCUDADevices() {
    int count;
    cudaGetDeviceCount(&count);
    return count;
}

class IPsecAESCUDAMatchTest : public ::testing::TestWithParam<int> {
protected:
    virtual void SetUp() {
        cudaSetDevice(GetParam());

        struct ipaddr_pair pair;
        struct aes_sa_entry *entry;
        unsigned char fake_iv[AES_BLOCK_SIZE] = {0};

        aes_sa_entry_array = (struct aes_sa_entry *) malloc(sizeof(struct aes_sa_entry) * num_tunnels);
        for (int i = 0; i < num_tunnels; i++) {
            pair.src_addr  = 0x0a000001u;
            pair.dest_addr = 0x0a000000u | (i + 1);
            auto result = aes_sa_table.insert(make_pair<ipaddr_pair&, int&>(pair, i));
            assert(result.second == true);

            entry = &aes_sa_entry_array[i];
            entry->entry_idx = i;
            rte_memcpy(entry->aes_key, "1234123412341234", AES_BLOCK_SIZE);
            #ifdef USE_OPENSSL_EVP
            EVP_CIPHER_CTX_init(&entry->evpctx);
            if (EVP_EncryptInit(&entry->evpctx, EVP_aes_128_ctr(), entry->aes_key, fake_iv) != 1)
                fprintf(stderr, "IPsecAES: EVP_EncryptInit() - %s\n", ERR_error_string(ERR_get_error(), NULL));
            #endif
            AES_set_encrypt_key((uint8_t *) entry->aes_key, 128, &entry->aes_key_t);
        }
    }

    virtual void TearDown() {
        free(aes_sa_entry_array);
        cudaDeviceReset();
    }

    const long num_tunnels = 1024;
    struct aes_sa_entry *aes_sa_entry_array;
    unordered_map<struct ipaddr_pair, int> aes_sa_table;
};

TEST_P(IPsecAESCUDAMatchTest, SingleBatchWithDatablock) {
}

INSTANTIATE_TEST_CASE_P(PerDeviceIPsecAESCUDAMatchTests, IPsecAESCUDAMatchTest,
                        ::testing::Values(0, getNumCUDADevices() - 1));

#endif

// vim: ts=8 sts=4 sw=4 et
