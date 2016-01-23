/*
 * bitmap.hh
 *
 *  Created on: 2014. 9. 12.
 *      Author: Keunhong Lee
 */

#ifndef __NBA_BITMAP_HPP__
#define __NBA_BITMAP_HPP__

#include <cstdint>
#include <cstddef>

namespace nba { }

namespace L
{

typedef uint64_t base_int;

class Bitmap
{
public:
	class Allocator
	{
	public:
		Allocator();
		virtual ~Allocator();
		virtual void* allocate(size_t size);
		virtual void deallocate(void* mem);
	};
private:
	Allocator* allocator = 0;
	base_int* array;
	size_t available_buckets;
	size_t size;
	static Allocator defaultAllocator;
public:
	Bitmap(size_t bits, Allocator* alloc = &defaultAllocator);
	Bitmap(const Bitmap& source);
	~Bitmap();

	bool isCollide(const Bitmap& other);
	void merge(const Bitmap& other);
	void intersect(const Bitmap& other);
	bool getBit(size_t index);
	void setBit(bool value, size_t index);
	void setRange(bool value, size_t start, size_t end);
	size_t getFirstBit();
	size_t getLastBit();
	void clear();
	void print(void);
};

}

#endif /* BITMAP_HPP_ */
