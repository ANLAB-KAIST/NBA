#ifndef __NBA_ELEMENT_IPSEC_IPSECAUTHHMACSHA1_KERNEL_HH__
#define __NBA_ELEMENT_IPSEC_IPSECAUTHHMACSHA1_KERNEL_HH__

#include <cuda.h>
#include <openssl/sha.h>
#include "util_sa_entry.hh"

#define HMAC_KEY_SIZE	 64

//----- not used in NBA currently. copied from the old NBA.
namespace nba {

enum {
	CUDA_THREADS_PER_HSHA1_BLK = 32,
	IP_ALIGNMENT_SIZE = 2 // Adjust start offset of ip header by shifting 2byte
};
//-----

extern void *ipsec_hsha1_encryption_get_cuda_kernel();

}
#endif /* __NBA_ELEMENT_IPSEC_IPSECAUTHHMACSHA1_KERNEL_HH__ */
