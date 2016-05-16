#ifndef __NBA_KNAPP_KERNELS_HH__
#define __NBA_KNAPP_KERNELS_HH__

namespace nba { namespace knapp {

enum kernel_types : uintptr_t {
    ID_KERNEL_IPV4LOOKUP = 1u,
    ID_KERNEL_IPV4LOOKUP_VECTOR = 2u,
    ID_KERNEL_IPV6LOOKUP = 3u,
    ID_KERNEL_IPV6LOOKUP_VECTOR = 3u,
    ID_KERNEL_IPSEC_AES = 4u,
    ID_KERNEL_IPSEC_HMACSHA1 = 5u,
};

#define KNAPP_MAX_KERNEL_TYPES (6u)

}} //endns(nba::knapp)

#endif //__NBA_KNAPP_KERNELS_HH__

// vim: ts=8 sts=4 sw=4 et
