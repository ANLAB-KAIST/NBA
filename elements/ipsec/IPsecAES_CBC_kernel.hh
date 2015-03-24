#ifndef __NBA_ELEMENT_IPSEC_IPSECAES_CBC_KERNEL_HH__
#define __NBA_ELEMENT_IPSEC_IPSECAES_CBC_KERNEL_HH__

#include <cuda.h>
#include <stdint.h>

#include "util_sa_entry.hh"

namespace nba {

extern void *ipsec_aes_encryption_cbc_get_cuda_kernel();

}
#endif /* __NBA_ELEMENT_IPSEC_IPSECAES_CBC_KERNEL_HH__ */
