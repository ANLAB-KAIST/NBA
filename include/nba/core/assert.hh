#ifndef __NBA_CORE_ASSERT_HH__
#define __NBA_CORE_ASSERT_HH__

#include <cstdio>
#include <cstdlib>

// Define an empty namespace for binaries with macro-only headers.
namespace nba { }

#define test_assert(cond, msg) {\
    if (!(cond)) { \
        fprintf(stderr, "Assertion failure (Line %d): %s, %s\n", __LINE__, #cond, msg); \
        exit(1); \
    } \
}

#endif

// vim: ts=8 sts=4 sw=4 et
