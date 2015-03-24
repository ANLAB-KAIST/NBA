/*
 * bitmap.cpp
 *
 *  Created on: 2014. 9. 12.
 *      Author: 근홍
 */




#include "bitmap.hpp"
#include <cstdlib>
#include <cmath>
#include <memory>
#include <cassert>
#include <cstring>

namespace L
{

#define BIT_COUNT ((size_t)(sizeof(base_int)*8)) //XXX replace it with sizeof(base_int)*8 after GCC bug is fixed
#define LOW_FLAG ((base_int)1)
#define HIGH_FLAG ((LOW_FLAG) << (BIT_COUNT-1))
#define ZERO ((base_int)0)

constexpr static base_int mark_bit(size_t high_index)
{
	return ((base_int)(HIGH_FLAG) >> (base_int)(high_index));
}

constexpr static inline base_int fill_low_bit(size_t count)
{
	return (base_int)((base_int)(~ZERO) >> (BIT_COUNT - count)) & (base_int)((base_int)(~ZERO) + (base_int)(!count));
}

constexpr static inline base_int fill_high_bit(size_t count)
{
	return (base_int)((base_int)(~ZERO) << (BIT_COUNT - count)) & (base_int)((base_int)(~ZERO) + (base_int)(!count));
}

constexpr static inline size_t bucket_index(size_t count)
{
	return count / BIT_COUNT;
}

constexpr static inline size_t sub_index(size_t count)
{
	return count % BIT_COUNT;
}

Bitmap::Allocator Bitmap::defaultAllocator;

inline Bitmap::Allocator::Allocator()
{

}

inline Bitmap::Allocator::~Allocator()
{

}

inline void* Bitmap::Allocator::allocate(size_t size)
{
	void* ptr = 0;
	int ret = posix_memalign(&ptr, sizeof(base_int), size);
	assert(ret == 0);
	memset(ptr, 0, size);
	return ptr;
}

inline void Bitmap::Allocator::deallocate(void* prev)
{
	return free(prev);
}

Bitmap::Bitmap(size_t bits, Allocator* alloc)
{
	this->allocator = alloc;
	this->array = 0;
	this->available_buckets = 0;


	this->available_buckets = (bits + BIT_COUNT - 1) / BIT_COUNT;
	size_t buffer_size = ((this->available_buckets) * sizeof(base_int));

	this->array = (base_int*)this->allocator->allocate(buffer_size);
	this->size = bits;
}
Bitmap::Bitmap(const Bitmap& source)
{
	{
		this->allocator = source.allocator;
		this->array = 0;
		this->available_buckets = 0;


		this->available_buckets = (source.size + BIT_COUNT - 1) / BIT_COUNT;
		size_t buffer_size = ((this->available_buckets) * sizeof(base_int));

		this->array = (base_int*)this->allocator->allocate(buffer_size);
		this->size = source.size;
	}
	for(size_t k=0; k<source.available_buckets; k++)
		this->array[k] = source.array[k];
}
Bitmap::~Bitmap()
{
	if(this->array)
	{
		this->allocator->deallocate(this->array);
		this->array = 0;
		this->available_buckets = 0;
		this->size = 0;
	}

	this->allocator = 0;
}

bool Bitmap::isCollide(const Bitmap& other)
{
	for(size_t k = 0; k<other.available_buckets && k<this->available_buckets; k++)
		if(this->array[k] & other.array[k])
			return true;
	return false;
}
void Bitmap::merge(const Bitmap& other)
{
	assert(this->size >= other.size);
	for(size_t k = 0; k<other.available_buckets; k++)
		this->array[k] |= other.array[k];
}
void Bitmap::intersect(const Bitmap& other)
{
	assert(this->size <= other.size);
	for(size_t k = 0; k<other.available_buckets; k++)
		this->array[k] &= other.array[k];
}

bool Bitmap::getBit(size_t index)
{
	return (bool)!!(this->array[bucket_index(index)] & mark_bit(sub_index(index)));
}
void Bitmap::setBit(bool value, size_t index)
{
	if(value)
		this->array[bucket_index(index)] |= mark_bit(sub_index(index));
	else
		this->array[bucket_index(index)] &= ~(mark_bit(sub_index(index)));
}
void Bitmap::setRange(bool value, size_t start, size_t end)
{
	size_t start_bucket = bucket_index(start);
	size_t end_bucket = bucket_index(end);
	size_t start_subindex = sub_index(start);
	size_t end_subindex = sub_index(end);

	base_int start_mask = fill_low_bit(BIT_COUNT - start_subindex);
	base_int end_mask = fill_high_bit(end_subindex);

	if(start_bucket == end_bucket)
	{
		base_int mask = start_mask & end_mask;
		if(value)
			this->array[start_bucket] |= mask;
		else
			this->array[start_bucket] &= ~mask;
	}
	else
	{
		if(value)
		{
			this->array[start_bucket] |= start_mask;
			if(end_mask)
				this->array[end_bucket] |= end_mask;
			for(size_t k = start_bucket+1; k<end_bucket; k++)
			{
				this->array[k] = (~ZERO);
			}
		}
		else
		{
			this->array[start_bucket] &= ~start_mask;
			if(end_mask)
				this->array[end_bucket] &= ~end_mask;
			for(size_t k = start_bucket+1; k<end_bucket; k++)
			{
				this->array[k] = (ZERO);
			}
		}
	}
}

size_t Bitmap::getFirstBit()
{
	for(size_t k = 0; k<this->available_buckets; k++)
	{
		base_int bucket = this->array[k];
		if(bucket != 0)
		{
			for(size_t sub = 0; sub < BIT_COUNT; sub++)
			{
				if(mark_bit(sub) & bucket)
				{
					return (k * BIT_COUNT) + sub;
				}
			}
		}
	}
	return 0;
}
size_t Bitmap::getLastBit()
{
	size_t k = this->available_buckets;
	while(1)
	{
		k--;
		base_int bucket = this->array[k];
		if(bucket != 0)
		{
			size_t sub = BIT_COUNT;
			while(1)
			{
				sub--;
				if(mark_bit(sub) & bucket)
				{
					return (k * BIT_COUNT) + sub;
				}

				if(sub == 0)
					break;
			}
		}

		if(k == 0)
			break;
	}
	return 0;
}

void Bitmap::clear()
{
	for(size_t k = 0; k<this->available_buckets; k++)
	{
		this->array[k] = 0;
	}
}
}

#include <iostream>
void L::Bitmap::print()
{
	for(size_t k=0; k<this->size; k++)
	{
		bool ret = this->getBit(k);
		std::cout<<ret;
	}
	std::cout<<std::endl;
}
