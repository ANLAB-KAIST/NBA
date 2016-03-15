#ifndef __NBA_IPSEC_SA_ENTRY_HH__
#define __NBA_IPSEC_SA_ENTRY_HH__

#include <openssl/aes.h>
#include <openssl/evp.h>

enum {
    // AES_BLOCK_SIZE is defined at openssl/aes.h
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
