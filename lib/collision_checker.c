/*
 * collision_checker.c
 *
 *  Created on: 2014. 5. 10.
 *      Author: leeopop
 */


#include <sys/stat.h>
#include <sys/file.h>
#include <stdint.h>
#include <limits.h>
#include <linux/limits.h>
#include <string.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include "collision_checker.h"

int check_collision(const char* program_name, uint32_t flags)
{
	char program[NAME_MAX];
	char dir[NAME_MAX];
	if(flags & COLLISION_USE_TEMP)
		strncpy(dir, "tmp", NAME_MAX);
	else
		strncpy(dir, "run", NAME_MAX);

	strncpy(program, program_name, NAME_MAX);

	char path[PATH_MAX];
	strcpy(path, "/");
	strcat(path, dir);
	strcat(path, "/");
	strcat(path, program);
	mkdir(path, 00755);

	strcat(path, "/");
	strcat(path, program);
	strcat(path, ".pid");

	int flock_flag = LOCK_EX;

	if(flags & COLLISION_NOWAIT)
	{
		flock_flag = flock_flag | LOCK_NB;
	}

	int success = -1;
	while(1)
	{
		pid_t pid = getpid();
		int fd = open(path, O_CREAT | O_SYNC | O_WRONLY, 00644);
		if(fd < 0)
		{
			fprintf(stderr, "Cannot open lock file\n");
			fprintf(stderr, "%d\n", remove(path));
			fflush(stderr);

			sleep(1);
			continue;
		}
		if(flock(fd, flock_flag) < 0)
		{
			close(fd);
			fprintf(stderr, "Someone is using lock file\n");
			fflush(stderr);
			break;
		}
		if(ftruncate(fd, 0) < 0)
		{
			fprintf(stderr, "truncate failed\n");
			fflush(stderr);
			break;
		}

		success = 0;
		char pid_buf[NAME_MAX];
		int len = snprintf(pid_buf, NAME_MAX, "%d\n", pid);
		write(fd, pid_buf, len);
		break;
	}

	return success;
}
