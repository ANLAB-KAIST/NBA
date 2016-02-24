#include <cassert>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include "ip_route_core.hh"

using namespace std;
using namespace nba;
using namespace nba::ipv4route;

int nba::ipv4route::add_route(
    route_hash_t *tables, uint32_t addr, uint16_t len, uint16_t nexthop)
{
    tables[len][addr] = nexthop;
    return 0;
}

int nba::ipv4route::delete_route(
    route_hash_t *tables, uint32_t addr, uint16_t len)
{
    tables[len].erase(addr);
    return 0;
}

int nba::ipv4route::load_rib_from_file(
    route_hash_t *tables, const char* filename)
{
    FILE *fp;
    char buf[256];

    fp = fopen(filename, "r");
    if (fp == NULL) {
        getcwd(buf, 256);
        printf("NBA: IpCPULookup element: error during opening file \'%s\' from \'%s\'.: %s\n", filename, buf, strerror(errno));
    }
    assert(fp != NULL);

    while (fgets(buf, 256, fp)) {
        char *str_addr = strtok(buf, "/");
        char *str_len = strtok(NULL, "\n");
        assert(str_len != NULL);

        uint32_t addr = ntohl(inet_addr(str_addr));
        uint16_t len = atoi(str_len);

        add_route(tables, addr, len, rand() % 65536);
    }
    fclose(fp);
    return 0;
}

int nba::ipv4route::build_direct_fib(
    const route_hash_t *tables, uint16_t *TBL24, uint16_t *TBLlong)
{
    // build_fib() is called for each node sequencially, before comp thread starts.
    // No rwlock protection is needed.
    memset(TBL24, 0, TBL24_SIZE * sizeof(uint16_t));
    memset(TBLlong, 0, TBLLONG_SIZE * sizeof(uint16_t));
    unsigned int current_TBLlong = 0;

    for (unsigned i = 0; i <= 24; i++) {
        for (auto it = tables[i].begin(); it != tables[i].end(); it++) {
            uint32_t addr = (*it).first;
            uint16_t dest = (uint16_t)(0xffffu & (uint64_t)(*it).second);
            uint32_t start = addr >> 8;
            uint32_t end = start + (0x1u << (24 - i));
            for (unsigned k = start; k < end; k++)
                TBL24[k] = dest;
        }
    }

    for (unsigned i = 25; i <= 32; i++) {
        for (auto it = tables[i].begin(); it != tables[i].end(); it++) {
            uint32_t addr = (*it).first;
            uint16_t dest = (uint16_t)(0x0000ffff & (uint64_t)(*it).second);
            uint16_t dest24 = TBL24[addr >> 8];
            if (((uint16_t)dest24 & 0x8000u) == 0) {
                uint32_t start = current_TBLlong + (addr & 0x000000ff);
                uint32_t end = start + (0x00000001u << (32 - i));

                for (unsigned j = current_TBLlong; j <= current_TBLlong + 256; j++)
                {
                    if (j < start || j >= end)
                        TBLlong[j] = dest24;
                    else
                        TBLlong[j] = dest;
                }
                TBL24[addr >> 8]  = (uint16_t)(current_TBLlong >> 8) | 0x8000u;
                current_TBLlong += 256;
                assert(current_TBLlong <= TBLLONG_SIZE);
            } else {
                uint32_t start = ((uint32_t)dest24 & 0x7fffu) * 256 + (addr & 0x000000ff);
                uint32_t end = start + (0x00000001u << (32 - i));

                for (unsigned j = start; j < end; j++)
                    TBLlong[j] = dest;
            }
        }
    }
    return 0;
}

void nba::ipv4route::direct_lookup(
    const uint16_t *TBL24, const uint16_t *TBLlong,
    const uint32_t ip, uint16_t *dest)
{
    uint16_t temp_dest;
    temp_dest = TBL24[ip >> 8];
    if (temp_dest & 0x8000u) {
        int index2 = (((uint32_t)(temp_dest & 0x7fff)) << 8) + (ip & 0xff);
        temp_dest = TBLlong[index2];
    }
    *dest = temp_dest;
}

// vim: ts=8 sts=4 sw=4 et
