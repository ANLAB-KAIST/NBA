#ifndef __NSHADER_ELEMENT_IPSEC_IPSECAES_KERNEL_HH__
#define __NSHADER_ELEMENT_IPSEC_IPSECAES_KERNEL_HH__

#include <cuda.h>
#include <stdint.h>

#include "util_sa_entry.hh"

namespace nshader {
enum {
	CUDA_THREADS_PER_AES_BLK = 256,
	MAX_ALIGNMENT_SIZE = 64,
	IP_ALIGNMENT_SIZE = 2	// Adjust start offset of ip header by shifting 2byte
};

extern void *ipsec_aes_encryption_get_cuda_kernel();

//#define PER_PKT_INFO

}
#endif /* __NSHADER_ELEMENT_IPSEC_IPSECAES_KERNEL_HH__ */
