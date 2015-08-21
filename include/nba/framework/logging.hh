#ifndef __NBA_LOGGING_HH__
#define __NBA_LOGGING_HH__

#include <rte_config.h>
#include <rte_log.h>

/* User-defined log types */
#define RTE_LOGTYPE_MAIN    RTE_LOGTYPE_USER1
#define RTE_LOGTYPE_IO      RTE_LOGTYPE_USER2
#define RTE_LOGTYPE_COMP    RTE_LOGTYPE_USER3
#define RTE_LOGTYPE_COPROC  RTE_LOGTYPE_USER4
#define RTE_LOGTYPE_ELEM    RTE_LOGTYPE_USER5
#define RTE_LOGTYPE_LB      RTE_LOGTYPE_USER6

#endif // __NBA_LOG_HH__

// vim: ts=8 sts=4 sw=4 et
