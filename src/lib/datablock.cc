#include <nba/framework/config.hh>
#include <nba/framework/datablock.hh>
#include <nba/element/element.hh>
#include <nba/element/packetbatch.hh>
#include <vector>
#include <string>
#include <cstring>
#include <rte_config.h>
#include <rte_memory.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_branch_prediction.h>

using namespace std;
using namespace nba;

namespace nba {

/*
 * NOTE: you may use STL vectors or other containers here,
 *       but in such cases you MUST GAURANTEE the build order
 *       to have lib/datablock.o earlier than datablock implementations.
 *       Otherwise, datablock.o may "re-instantiate" the containers and
 *       some implementations won't be registered.
 */
size_t num_datablocks = 0;
const char* datablock_names[NBA_MAX_DATABLOCKS];
datablock_constructor datablock_ctors[NBA_MAX_DATABLOCKS];

void declare_datablock_impl(const char *name, datablock_constructor ctor, int &your_id) {
    if (num_datablocks == NBA_MAX_DATABLOCKS)
        rte_panic("Too many datablock implementations!\n");
    for (unsigned i = 0; i < num_datablocks; i++) {
        if (strcmp(datablock_names[i], name) == 0) {
            your_id = i;
            return;
        }
    }
    your_id = num_datablocks ++;
    datablock_names[your_id] = name;
    datablock_ctors[your_id] = ctor;
}

tuple<size_t, size_t> DataBlock::calc_read_buffer_size(PacketBatch *batch)
{
    read_buffer_size = 0;
    size_t num_read_items = 0;
    size_t accum_idx = 0;
    struct read_roi_info read_roi;
    this->get_read_roi(&read_roi);
    if (this->merged_datablock_idx != -1)
        read_roi.offset = this->merged_read_offset;

    struct datablock_tracker *t = &batch->datablock_states[this->get_id()];

    switch (read_roi.type) {
    case READ_PARTIAL_PACKET: {

        /* Copy a portion of packets or user-define fixed-size values.
         * We use a fixed-size range (offset, length) here.
         * The buffer is NOT aligned unless the element explicitly
         * specifies the alignment. */
        num_read_items = batch->count;
        size_t align = (read_roi.align == 0) ? 2 : read_roi.align;
        unsigned aligned_len = RTE_ALIGN_CEIL(read_roi.length, align);
        t->aligned_item_sizes->size = aligned_len;
        read_buffer_size            = aligned_len * num_read_items;

        break; }
    case READ_WHOLE_PACKET: {

        /* Copy the whole content of packets.
         * We align the buffer by the cache line size,
         * or the alignment explicitly set by the element. */

        num_read_items = batch->count;
        size_t align = (read_roi.align == 0) ? CACHE_LINE_SIZE : read_roi.align;
        FOR_EACH_PACKET_ALL(batch) {
            #if (NBA_BATCHING_SCHEME == NBA_BATCHING_TRADITIONAL) \
                || (NBA_BATCHING_SCHEME == NBA_BATCHING_BITVECTOR)
            if (IS_PACKET_INVALID(batch, pkt_idx)) {
                t->aligned_item_sizes->offsets[pkt_idx] = 0;
                t->aligned_item_sizes->sizes[pkt_idx]   = 0;
            } else {
            #endif
                unsigned exact_len   = rte_pktmbuf_data_len(batch->packets[pkt_idx]) - read_roi.offset
                                       + read_roi.length + read_roi.size_delta;
                unsigned aligned_len = RTE_ALIGN_CEIL(exact_len, align);
                t->aligned_item_sizes->offsets[pkt_idx] = read_buffer_size;
                t->aligned_item_sizes->sizes[pkt_idx]   = exact_len;
                read_buffer_size += aligned_len;
            #if (NBA_BATCHING_SCHEME == NBA_BATCHING_TRADITIONAL) \
                || (NBA_BATCHING_SCHEME == NBA_BATCHING_BITVECTOR)
            } /* endif(excluded) */
            #endif
        } END_FOR_ALL;

        break; }
    case READ_USER_PREPROC: {

        /* NOTE: We assume read_buffer_size is multiple of num_read_items. */
        this->calculate_read_buffer_size(batch, read_buffer_size, num_read_items);

        break; }
    case READ_NONE: {

        read_buffer_size = 0;
        num_read_items = 0;

        break; }
    default:

        rte_panic("DataBlock::calc_read_buffer_size(): Unsupported read_roi.\n");

        break;
    }
    // TODO: align again (e.g., page boundary)?
    return make_tuple(read_buffer_size, num_read_items);
}

tuple<size_t, size_t> DataBlock::calc_write_buffer_size(PacketBatch *batch)
{
    write_buffer_size = 0;
    size_t num_write_items = 0;
    struct read_roi_info read_roi;
    struct write_roi_info write_roi;
    this->get_read_roi(&read_roi);
    this->get_write_roi(&write_roi);
    if (this->merged_datablock_idx != -1)
        write_roi.offset = this->merged_write_offset;

    struct datablock_tracker *t = &batch->datablock_states[this->get_id()];

    switch (write_roi.type) {
    case WRITE_PARTIAL_PACKET: {

        size_t align         = (write_roi.align == 0) ? 2 : write_roi.align;
        unsigned aligned_len = RTE_ALIGN_CEIL(write_roi.length, align);
        num_write_items   = batch->count;
        write_buffer_size = aligned_len * num_write_items;

        break; }
    case WRITE_WHOLE_PACKET: {

        // FIXME: We currently assume same-as-input when read_roi.type is same.
        assert(read_roi.type == READ_WHOLE_PACKET && write_roi.type == WRITE_WHOLE_PACKET);
        num_write_items   = batch->count;
        write_buffer_size = read_buffer_size;

        break; }
    case WRITE_FIXED_SEGMENTS: {

        num_write_items = batch->count;
        write_buffer_size     = write_roi.length * num_write_items;

        break; }
    case WRITE_USER_POSTPROC: {

        break; }
    case WRITE_NONE: {

        break; }
    default:

        rte_panic("DataBlock::calc_write_buffer_size(): Unsupported write_roi.\n");

        break;
    }
    return make_tuple(write_buffer_size, num_write_items);
}

void DataBlock::preprocess(PacketBatch *batch, void *host_in_buffer) {
    struct read_roi_info read_roi;
    this->get_read_roi(&read_roi);
    if (this->merged_datablock_idx != -1)
        read_roi.offset = this->merged_read_offset;

    struct datablock_tracker *t = &batch->datablock_states[this->get_id()];

    switch (read_roi.type) {
    case READ_PARTIAL_PACKET: {
        void *invalid_value = this->get_invalid_value();
        FOR_EACH_PACKET_ALL_PREFETCH(batch, 4u) {
            uint16_t aligned_elemsz = t->aligned_item_sizes->size;
            uint32_t offset         = t->aligned_item_sizes->size * pkt_idx;
            if (IS_PACKET_INVALID(batch, pkt_idx)) {
                if (invalid_value != nullptr) {
                    rte_memcpy((char *) host_in_buffer + offset, invalid_value, aligned_elemsz);
                }
            } else {
                rte_memcpy((char*) host_in_buffer + offset,
                           rte_pktmbuf_mtod(batch->packets[pkt_idx], char*) + read_roi.offset,
                           aligned_elemsz);
            }
        } END_FOR_ALL_PREFETCH;

        break; }
    case READ_WHOLE_PACKET: {

        /* Copy the speicified region of packet to the input buffer. */
        FOR_EACH_PACKET_ALL_PREFETCH(batch, 4u) {
            if (IS_PACKET_INVALID(batch, pkt_idx))
                continue;
            size_t aligned_elemsz = t->aligned_item_sizes->sizes[pkt_idx];
            size_t offset         = t->aligned_item_sizes->offsets[pkt_idx].as_value<size_t>();
            rte_memcpy((char*) host_in_buffer + offset,
                       rte_pktmbuf_mtod(batch->packets[pkt_idx], char*) + read_roi.offset,
                       aligned_elemsz);
        } END_FOR_ALL_PREFETCH;

        break; }
    case READ_USER_PREPROC: {

        preproc_batch(batch, host_in_buffer);

        break; }
    case READ_NONE: {

        break; }
    default:

        rte_panic("DataBlock::preprocess(): Unsupported read_roi.\n");
        break;
    }
}

void DataBlock::postprocess(OffloadableElement *elem, int input_port, PacketBatch *batch, void *host_out_ptr)
{
    struct write_roi_info write_roi;
    this->get_write_roi(&write_roi);
    if (this->merged_datablock_idx != -1)
        write_roi.offset = this->merged_write_offset;

    struct datablock_tracker *t = &batch->datablock_states[this->get_id()];

    // FIXME: 현재는 write, 즉 output이 있는 datablock이 element 당 1개만 있다고 생각.
    //        만약 여러 개 있을 경우 postproc() 메소드가 여러 번 불리게 된다.

    switch (write_roi.type) {
    case WRITE_PARTIAL_PACKET:
    case WRITE_WHOLE_PACKET: {

        /* Update the packets and run postprocessing. */
        #if NBA_BATCHING_SCHEME == NBA_BATCHING_CONTINUOUS
        batch->has_dropped = false;
        #endif
        FOR_EACH_PACKET(batch) {
            size_t elemsz = bitselect<size_t>(write_roi.type == WRITE_PARTIAL_PACKET,
                                              t->aligned_item_sizes->size,
                                              t->aligned_item_sizes->sizes[pkt_idx]);
            size_t offset = bitselect<size_t>(write_roi.type == WRITE_PARTIAL_PACKET,
                                              t->aligned_item_sizes->size * pkt_idx,
                                              t->aligned_item_sizes->offsets[pkt_idx].as_value<size_t>());
            rte_memcpy(rte_pktmbuf_mtod(batch->packets[pkt_idx], char*) + write_roi.offset,
                       (char*) host_out_ptr + offset,
                       elemsz);
            Packet *pkt = Packet::from_base(batch->packets[pkt_idx]);
            pkt->bidx = pkt_idx;
            elem->postproc(input_port, nullptr, pkt);
        } END_FOR;
        #if NBA_BATCHING_SCHEME == NBA_BATCHING_CONTINUOUS
        if (batch->has_dropped)
            batch->collect_excluded_packets();
        #endif
        batch->tracker.has_results = true;

        break; }
    case WRITE_USER_POSTPROC:

        postproc_batch(batch, host_out_ptr);

    case WRITE_FIXED_SEGMENTS: {

        /* Run postporcessing only. */
        #if NBA_BATCHING_SCHEME == NBA_BATCHING_CONTINUOUS
        batch->has_dropped = false;
        #endif
        FOR_EACH_PACKET(batch) {
            uintptr_t elemsz = write_roi.length;
            uintptr_t offset = elemsz * pkt_idx;
            Packet *pkt = Packet::from_base(batch->packets[pkt_idx]);
            pkt->bidx = pkt_idx;
            elem->postproc(input_port, (char*) host_out_ptr + offset, pkt);
        } END_FOR;
        #if NBA_BATCHING_SCHEME == NBA_BATCHING_CONTINUOUS
        if (batch->has_dropped)
            batch->collect_excluded_packets();
        #endif
        batch->tracker.has_results = true;

        break; }
    case WRITE_NONE: {

        break; }
    default:

        rte_panic("DataBlock::postprocess(): Unsupported write_roi.\n");

        break;
    }
}

}

// vim: ts=8 sts=4 sw=4 et
