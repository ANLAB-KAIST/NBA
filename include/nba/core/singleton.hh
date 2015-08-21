/*
 * singleton.h
 *
 * It enforces only one NBA process can proceed in the system.
 *
 *  Created on: 2014. 5. 10.
 *      Author: leeopop
 */

#ifndef __NBA_SINGLETON_HH__
#define __NBA_SINGLETON_HH__

#include <cstdint>

#define COLLISION_NOWAIT (1)
#define COLLISION_USE_TEMP (2)

namespace nba {

int check_collision(const char* program_name, uint32_t flags);

}

#endif /* __NBA_SINGLETON_HH__ */
