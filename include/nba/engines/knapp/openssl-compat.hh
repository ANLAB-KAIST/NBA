#ifndef __NBA_KNAPP_OPENSSL_COMPAT_HH__
#define __NBA_KNAPP_OPENSSL_COMPAT_HH__

/* This header is to compile IPsec utility headers for Intel Xeon Phi
 * without cross-compiling the whole OpenSSL library. */

#ifndef __MIC__
#error "This header must be used for MIC only."
#endif


extern "C" {

// from openssl/aes.h

#define AES_ENCRYPT     1
#define AES_DECRYPT     0

/* Because array size can't be a const in C, the following two are macros.
   Both sizes are in bytes. */
#define AES_MAXNR 14
#define AES_BLOCK_SIZE 16

/* This should be a hidden type, but EVP requires that the size be known */
struct aes_key_st {
#ifdef AES_LONG
    unsigned long rd_key[4 *(AES_MAXNR + 1)];
#else
    unsigned int rd_key[4 *(AES_MAXNR + 1)];
#endif
    int rounds;
};
typedef struct aes_key_st AES_KEY;


// from openssl/sha.h

#define SHA_LONG unsigned int

#define SHA_LBLOCK      16
#define SHA_CBLOCK      (SHA_LBLOCK*4)  /* SHA treats input data as a
                                         * contiguous array of 32 bit
                                         * wide big-endian values. */
#define SHA_LAST_BLOCK  (SHA_CBLOCK-8)
#define SHA_DIGEST_LENGTH 20

typedef struct SHAstate_st
    {
        SHA_LONG h0,h1,h2,h3,h4;
        SHA_LONG Nl,Nh;
        SHA_LONG data[SHA_LBLOCK];
        unsigned int num;
    } SHA_CTX;


// from openssl/ossl_typ.h

typedef struct evp_cipher_ctx_st EVP_CIPHER_CTX;
typedef struct evp_cipher_st EVP_CIPHER;


// from openssl/asn1.h

typedef struct asn1_type_st
    {
        int type;
        union   {
            void *whatever_asntype_pointer;
        } value;
    } ASN1_TYPE;


// from openssl/evp.h

#define EVP_MAX_IV_LENGTH               16
#define EVP_MAX_BLOCK_LENGTH            32

struct evp_cipher_ctx_st
    {
        const EVP_CIPHER *cipher;
        //ENGINE *engine;       /* functional reference if 'cipher' is ENGINE-provided */
        void *engine;
        int encrypt;            /* encrypt or decrypt */
        int buf_len;            /* number we have left */

        unsigned char  oiv[EVP_MAX_IV_LENGTH];  /* original iv */
        unsigned char  iv[EVP_MAX_IV_LENGTH];   /* working iv */
        unsigned char buf[EVP_MAX_BLOCK_LENGTH];/* saved partial block */
        int num;                                /* used by cfb/ofb/ctr mode */

        void *app_data;         /* application stuff */
        int key_len;            /* May change for variable length cipher */
        unsigned long flags;    /* Various flags */
        void *cipher_data; /* per EVP data */
        int final_used;
        int block_mask;
        unsigned char final[EVP_MAX_BLOCK_LENGTH];/* possible final block */
    } /* EVP_CIPHER_CTX */;

struct evp_cipher_st
    {
        int nid;
        int block_size;
        int key_len;            /* Default value for variable length ciphers */
        int iv_len;
        unsigned long flags;    /* Various flags */
        int (*init)(EVP_CIPHER_CTX *ctx, const unsigned char *key,
                    const unsigned char *iv, int enc);  /* init key */
        int (*do_cipher)(EVP_CIPHER_CTX *ctx, unsigned char *out,
                         const unsigned char *in, size_t inl);/* encrypt/decrypt data */
        int (*cleanup)(EVP_CIPHER_CTX *); /* cleanup ctx */
        int ctx_size;           /* how big ctx->cipher_data needs to be */
        int (*set_asn1_parameters)(EVP_CIPHER_CTX *, ASN1_TYPE *); /* Populate a ASN1_TYPE with parameters */
        int (*get_asn1_parameters)(EVP_CIPHER_CTX *, ASN1_TYPE *); /* Get parameters from a ASN1_TYPE */
        int (*ctrl)(EVP_CIPHER_CTX *, int type, int arg, void *ptr); /* Miscellaneous operations */
        void *app_data;         /* Application data */
    } /* EVP_CIPHER */;

} // endextern("C")


#endif //__NBA_KNAPP_OPENSSL_COMPAT_HH__

// vim: ts=8 sts=4 sw=4 et
