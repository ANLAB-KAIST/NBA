#ifndef __NSHADER_ELEMENT_IPSEC_IPSECAES_CBC_KERNEL_HH__
#define __NSHADER_ELEMENT_IPSEC_IPSECAES_CBC_KERNEL_HH__

#include <cuda.h>
#include <stdint.h>

#include "util_sa_entry.hh"

namespace nshader {

extern void *ipsec_aes_encryption_cbc_get_cuda_kernel();

}
#endif /* __NSHADER_ELEMENT_IPSEC_IPSECAES_CBC_KERNEL_HH__ */
