#ifndef __NBA_CORE_VECTOR_HH__
#define __NBA_CORE_VECTOR_HH__

extern "C" {
#include <immintrin.h>
}
#include <cstdint>

/* FIXME: support different vector widths by detecting the current arch. */

#define NBA_VECTOR_WIDTH (8)  /* in 4-bytes unit */
/* The reason to use 4-bytes unit is that scatter/gather operations
 * uses a special addressing scheme: 32-bit integers added to a base
 * address. */

typedef __m256i vec_mask_t;

/* Vectors are registers; we cannot use them as function args directly. */
typedef struct { uint32_t m[NBA_VECTOR_WIDTH]; } vec_mask_arg_t;

#endif // __NBA_CORE_VECTOR_HH__

// vim: ts=8 sts=4 sw=4 et
