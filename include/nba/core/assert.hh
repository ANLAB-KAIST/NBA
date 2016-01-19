#ifndef __NBA_CORE_ASSERT_HH__
#define __NBA_CORE_ASSERT_HH__

#include <cstdio>
#include <cstdlib>

#define test_assert(cond, msg) {\
    if (!(cond)) { \
        fprintf(stderr, "Assertion failure: %s, %s\n", #cond, msg); \
        exit(1); \
    } \
}

#endif

// vim: ts=8 sts=4 sw=4 et
