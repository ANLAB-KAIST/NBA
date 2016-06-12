#ifndef __NBA_IPSEC_SA_ENTRY_HH__
#define __NBA_IPSEC_SA_ENTRY_HH__

#ifdef __MIC__
#include <nba/engines/knapp/openssl-compat.hh>
#else
#include <openssl/aes.h>
#include <openssl/evp.h>
#endif

#ifdef AES_BLOCK_SIZE
#undef AES_BLOCK_SIZE
#endif
enum : int {
    AES_BLOCK_SIZE = 16,
    HMAC_KEY_SIZE = 64,
};

struct alignas(8) aes_block_info {
    int pkt_idx;
    int block_idx;
    int pkt_offset;
    int magic;
};

struct alignas(8) aes_sa_entry {
    // Below two variables have same value.
    uint8_t aes_key[AES_BLOCK_SIZE];    // Used in CUDA encryption.
    AES_KEY aes_key_t;                  // Prepared for AES library function.
    EVP_CIPHER_CTX evpctx;
    int entry_idx;                      // Index of current flow: value for verification.
};

struct alignas(8) hmac_sa_entry {
    uint8_t hmac_key[HMAC_KEY_SIZE];
    int entry_idx;
};

struct alignas(8) hmac_aes_sa_entry {
    // Below two variables have same value.
    uint8_t aes_key[AES_BLOCK_SIZE];    // Used in CUDA encryption.
    AES_KEY aes_key_t;                  // Prepared for AES library function.
    EVP_CIPHER_CTX evpctx;
    int entry_idx;                      // Index of current flow: value for varification.
    uint8_t hmac_key[HMAC_KEY_SIZE];
};

#endif

// vim: set ts=8 sts=4 sw=4 et
