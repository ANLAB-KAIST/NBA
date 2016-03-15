#ifndef __NBA_CORE_ERRORS_HH__
#define __NBA_CORE_ERRORS_HH__

namespace nba {

typedef enum : int {
    NBA_SUCCESS = 0,
    NBA_FAILURE = 1,
    NBA_NOT_FOUND = 2,
} error_t;

} // endns(nba)

#endif

// vim: ts=8 sts=4 sw=4 et
