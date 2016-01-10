#include "IPsecDatablocks.hh"
#include <rte_malloc.h>

namespace nba {

int dbid_enc_payloads;
int dbid_flow_ids;
int dbid_iv;
int dbid_aes_block_info;

static DataBlock* db_enc_payloads_ctor (void) {
    DataBlock *ptr = (DataBlock *) rte_malloc("datablock", sizeof(IPsecEncryptedPayloadDataBlock), CACHE_LINE_SIZE);
    assert(ptr != nullptr);
    new (ptr) IPsecEncryptedPayloadDataBlock();
    return ptr;
};
static DataBlock* db_flow_ids_ctor (void) {
    DataBlock *ptr = (DataBlock *) rte_malloc("datablock", sizeof(IPsecFlowIDsDataBlock), CACHE_LINE_SIZE);
    assert(ptr != nullptr);
    new (ptr) IPsecFlowIDsDataBlock();
    return ptr;
};
static DataBlock* db_iv_ctor (void) {
    DataBlock *ptr = (DataBlock *) rte_malloc("datablock", sizeof(IPsecIVDataBlock), CACHE_LINE_SIZE);
    assert(ptr != nullptr);
    new (ptr) IPsecIVDataBlock();
    return ptr;
};
static DataBlock* db_aes_block_info_ctor (void) {
    DataBlock *ptr = (DataBlock *) rte_malloc("datablock", sizeof(IPsecAESBlockInfoDataBlock), CACHE_LINE_SIZE);
    assert(ptr != nullptr);
    new (ptr) IPsecAESBlockInfoDataBlock();
    return ptr;
};

declare_datablock("ipsec.enc_payloads", db_enc_payloads_ctor, dbid_enc_payloads);
declare_datablock("ipsec.flow_ids", db_flow_ids_ctor, dbid_flow_ids);
declare_datablock("ipsec.iv", db_iv_ctor, dbid_iv);
declare_datablock("ipsec.aes_block_info", db_aes_block_info_ctor, dbid_aes_block_info);

}

// vim: ts=8 sts=4 sw=4 et
