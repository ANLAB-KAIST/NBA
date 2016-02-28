#include <cstdint>
#include <cstdlib>
#include <cstdio>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
#include <nba/framework/datablock.hh>
#include <nba/framework/datablock_shared.hh>
#include <nba/element/annotation.hh>
#include <nba/element/packet.hh>
#include <nba/element/packetbatch.hh>
#include <nba/framework/test_utils.hh>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <rte_mbuf.h>
#include "../elements/ipsec/util_esp.hh"
#include "../elements/ipsec/util_ipsec_key.hh"
#include "../elements/ipsec/util_sa_entry.hh"
#include "../elements/ipsec/IPsecAES_kernel.hh"
#include "../elements/ipsec/IPsecAuthHMACSHA1_kernel.hh"
#include "../elements/ipsec/IPsecDatablocks.hh"
#if 0
#require <lib/datablock.o>
#require <lib/test_utils.o>
#require "../elements/ipsec/IPsecAES_kernel.o"
#require "../elements/ipsec/IPsecAuthHMACSHA1_kernel.o"
#require "../elements/ipsec/IPsecDatablocks.o"
#endif

using namespace std;
using namespace nba;

TEST(IPsecTest, Loading) {
    EXPECT_TRUE(1);
}

// vim: ts=8 sts=4 sw=4 et
