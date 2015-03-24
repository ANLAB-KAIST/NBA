#ifndef __NBA_ELEMENT_IPSEC_IPSECHMACSHA1AES_KERNEL_HH__
#define __NBA_ELEMENT_IPSEC_IPSECHMACSHA1AES_KERNEL_HH__

#include <cuda.h>
#include <stdint.h>
#include <openssl/sha.h>

#include "util_sa_entry.hh"

namespace nba {

extern void *ipsec_hmac_sha1_aes_get_cuda_kernel();

}
#endif /* __NBA_ELEMENT_IPSEC_IPSECHMACSHA1AES_KERNEL_HH__ */
