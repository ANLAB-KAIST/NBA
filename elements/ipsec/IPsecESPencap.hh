#ifndef __NSHADER_ELEMENT_IPSEC_IPSECESPENCAP_HH__
#define __NSHADER_ELEMENT_IPSEC_IPSECESPENCAP_HH__


#include <rte_config.h>
#include <rte_memory.h>
#include <rte_mbuf.h>
#include <rte_ether.h>

#include "../../lib/element.hh"
#include "../../lib/annotation.hh"
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/ip.h>

#include <openssl/aes.h>
#include <openssl/sha.h>

#include "util_esp.hh"
#include "../ipv6/util_hash_table.hh"
#include "util_ipsec_key.hh"

namespace nshader {

class IPsecESPencap : public Element {
public:
	IPsecESPencap(): Element()
	{
		num_tunnels = 0;
		tunnel_counter = 0;
	}

	~IPsecESPencap()
	{
		struct espencap_sa_entry *sa_entry;

		for (auto iter = sa_table.begin(); iter != sa_table.end(); iter++) {
			auto sa_item = *iter;
			sa_entry = sa_item.second;
			delete sa_entry;
		}
	}

	const char *class_name() const { return "IPsecESPencap"; }
	const char *port_count() const { return "1/1"; }

	int initialize();
	int initialize_global()   { return 0; };	// per-system configuration
	int initialize_per_node() { return 0; };	// per-node configuration
	int configure(comp_thread_context *ctx, std::vector<std::string> &args);

	int process(int input_port, struct rte_mbuf *pkt, struct annotation_set *anno);

private:
	struct espencap_sa_entry {
		uint32_t spi;		/* Security Parameters Index */
		uint32_t rpl;		/* Replay counter */			// XXX: is this right to use this one?
		uint32_t gwaddr;	// XXX: not used yet; when this value is used?
		uint64_t entry_idx;
	};

	/* Maximum number of IPsec tunnels */
	int num_tunnels;

	/* Hash table which stores per-flow values for each tunnel */
	std::unordered_map<struct ipaddr_pair, struct espencap_sa_entry *> sa_table;

	/* A random function. */
	std::function<uint64_t()> rand;

	/* A temporary hack to allow all flows to be processed. */
	struct espencap_sa_entry *sa_table_linear[1024];
	uint64_t tunnel_counter;
};

EXPORT_ELEMENT(IPsecESPencap);

}

#endif
