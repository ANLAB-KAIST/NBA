#ifndef __NBA_KNAPP_KERNELS_HH__
#define __NBA_KNAPP_KERNELS_HH__

namespace nba { namespace knapp {

enum kernel_types : uintptr_t {
    ID_KERNEL_IPV4LOOKUP = 1u,
    ID_KERNEL_IPV6LOOKUP = 2u,
    ID_KERNEL_IPSEC_AES = 3u,
    ID_KERNEL_IPSEC_HMACSHA1 = 4u,
};

}} //endns(nba::knapp)

#endif //__NBA_KNAPP_KERNELS_HH__

// vim: ts=8 sts=4 sw=4 et
