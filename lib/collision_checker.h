/*
 * collision_checker.h
 *
 *  Created on: 2014. 5. 10.
 *      Author: leeopop
 */

#ifndef __NBA_COLLISION_CHECKER_H__
#define __NBA_COLLISION_CHECKER_H__

#include <stdint.h>

#define COLLISION_NOWAIT (1)
#define COLLISION_USE_TEMP (2)

#ifdef __cplusplus
extern "C" {
#endif
int check_collision(const char* program_name, uint32_t flags);
#ifdef __cplusplus
}
#endif

#endif /* __NBA_COLLISION_CHECKER_H__ */
