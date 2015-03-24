/*
 * ipsec_key.hh
 *
 *  Created on: Nov 21, 2011
 *      Author: seonggu
 */

#ifndef IPSEC_KEY_HH_
#define IPSEC_KEY_HH_

#include <stdint.h>
#include <unordered_map>

/**
 * IPv4 address pair to be used as hash-table keys.
 */
struct ipaddr_pair {
	uint32_t src_addr;
	uint32_t dest_addr;

	bool operator==(const ipaddr_pair &other) const
	{
		return (src_addr == other.src_addr && dest_addr == other.dest_addr);
	}
};

/* We need to define custom hash function for our key.
 * Just borrow the hash function for 64-bit integer as the key is a simple
 * pair of two 32-bit integers. */
namespace std {
	template<>
	struct hash<ipaddr_pair>
	{
	public:
		std::size_t operator()(ipaddr_pair const& p) const
		{
			return std::hash<uint64_t>()((((uint64_t)p.src_addr) << 32) | (p.dest_addr));
		}
	};
}

#endif /* IPSEC_KEY_HH_ */
