#include "datablock.hh"
#include "packetbatch.hh"
#include "element.hh"
#include "config.hh"
#include <vector>
#include <string>
#include <rte_config.h>
#include <rte_memory.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_branch_prediction.h>
#include <cstring>

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
    size_t read_buffer_size = 0, num_read_items = 0;
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
        t->aligned_item_sizes.size = aligned_len;
        read_buffer_size           = aligned_len * num_read_items;

        break; }
    case READ_WHOLE_PACKET: {

        /* Copy the whole content of packets.
         * We align the buffer by the cache line size,
         * or the alignment explicitly set by the element. */

        num_read_items = batch->count;
        size_t align = (read_roi.align == 0) ? CACHE_LINE_SIZE : read_roi.align;
        for (unsigned p = 0; p < batch->count; p++) {
            if (batch->excluded[p]) {
                t->aligned_item_sizes.offsets[p] = 0;
                //t->exact_item_sizes.sizes[p]   = 0;
                t->aligned_item_sizes.sizes[p]   = 0;
            } else {
                unsigned exact_len   = rte_pktmbuf_data_len(batch->packets[p]) - read_roi.offset
                                       + read_roi.length + read_roi.size_delta;
                unsigned aligned_len = RTE_ALIGN_CEIL(exact_len, align);
                t->aligned_item_sizes.offsets[p] = read_buffer_size;
                //t->exact_item_sizes.sizes[p]   = exact_len;
                t->aligned_item_sizes.sizes[p]   = aligned_len;
                read_buffer_size += aligned_len;
            }
        }

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
    size_t write_buffer_size = 0, num_write_items = 0;
    struct read_roi_info read_roi;
    struct write_roi_info write_roi;
    this->get_read_roi(&read_roi);
    this->get_write_roi(&write_roi);
    if (this->merged_datablock_idx != -1)
        write_roi.offset = this->merged_write_offset;

    struct datablock_tracker *t = &batch->datablock_states[this->get_id()];

    switch (write_roi.type) {
    case WRITE_PARTIAL_PACKET:
    case WRITE_WHOLE_PACKET: {

        num_write_items = batch->count;
        // FIXME: We currently assume same-as-input when read_roi.type is same.
        assert((read_roi.type == READ_PARTIAL_PACKET
                && write_roi.type == WRITE_PARTIAL_PACKET) ||
               (read_roi.type == READ_WHOLE_PACKET
                && write_roi.type == WRITE_WHOLE_PACKET));
        //if (read_roi.type == READ_PARTIAL_PACKET
        //        && write_roi.type == WRITE_PARTIAL_PACKET) {
        //    //write_item_sizes.size = aligned_item_sizes.size;
        //} else if (read_roi.type == READ_WHOLE_PACKET
        //        && write_roi.type == WRITE_WHOLE_PACKET) {
        //    //rte_memcpy(&write_item_sizes, &aligned_item_sizes,
        //    //           sizeof(struct item_size_info));
        //} else {
        //    assert(0); // Not implemented yet!
        //}
        write_buffer_size     = write_roi.length * num_write_items;

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
        #define PREFETCH_MAX (4)
        #if PREFETCH_MAX
        for (signed p = 0; p < RTE_MIN(PREFETCH_MAX, ((signed)batch->count)); p++)
            if (batch->packets[p] != nullptr)
                rte_prefetch0(rte_pktmbuf_mtod(batch->packets[p], void*));
        #endif
        void *invalid_value = this->get_invalid_value();
        for (unsigned p = 0; p < batch->count; p++) {
            size_t aligned_elemsz = t->aligned_item_sizes.size;
            size_t offset         = t->aligned_item_sizes.size * p;
            if (batch->excluded[p]) {
                if (invalid_value != nullptr) {
                    rte_memcpy((char *) host_in_buffer + offset, invalid_value, aligned_elemsz);
                }
            } else {
                #if PREFETCH_MAX
                if ((signed)p < (signed)batch->count - PREFETCH_MAX && batch->excluded[p + PREFETCH_MAX] == false)
                    rte_prefetch0(rte_pktmbuf_mtod(batch->packets[p + PREFETCH_MAX], void*));
                #endif
                rte_memcpy((char*) host_in_buffer + offset,
                           rte_pktmbuf_mtod(batch->packets[p], char*) + read_roi.offset,
                           aligned_elemsz);
            }
        }
        #undef PREFETCH_MAX

        break; }
    case READ_WHOLE_PACKET: {

        /* Copy the speicified region of packet to the input buffer. */
        #define PREFETCH_MAX (4)
        #if PREFETCH_MAX
        for (signed p = 0; p < RTE_MIN(PREFETCH_MAX, ((signed)batch->count)); p++)
            if (batch->packets[p] != nullptr)
                rte_prefetch0(rte_pktmbuf_mtod(batch->packets[p], void*));
        #endif
        for (unsigned p = 0; p < batch->count; p++) {
            if (batch->excluded[p])
                continue;
            size_t aligned_elemsz = t->aligned_item_sizes.sizes[p];
            size_t offset         = t->aligned_item_sizes.offsets[p];
            #if PREFETCH_MAX
            if ((signed)p < (signed)batch->count - PREFETCH_MAX && batch->excluded[p + PREFETCH_MAX] == false)
                rte_prefetch0(rte_pktmbuf_mtod(batch->packets[p + PREFETCH_MAX], void*));
            #endif
            rte_memcpy((char*) host_in_buffer + offset,
                       rte_pktmbuf_mtod(batch->packets[p], char*) + read_roi.offset,
                       aligned_elemsz);
        }
        #undef PREFETCH_MAX

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
        for (unsigned p = 0; p < batch->count; p++) {
            if (batch->excluded[p]) continue;
            size_t elemsz = bitselect<size_t>(write_roi.type == WRITE_PARTIAL_PACKET,
                                              t->aligned_item_sizes.size,
                                              t->aligned_item_sizes.sizes[p]);
            size_t offset = bitselect<size_t>(write_roi.type == WRITE_PARTIAL_PACKET,
                                              t->aligned_item_sizes.size * p,
                                              t->aligned_item_sizes.offsets[p]);
            rte_memcpy(rte_pktmbuf_mtod(batch->packets[p], char*) + write_roi.offset,
                       (char*) host_out_ptr + offset,
                       elemsz);
            Packet *pkt = Packet::from_base(batch->packets[p]);
            batch->results[p] = elem->postproc(input_port, nullptr, pkt);
            batch->excluded[p] = (batch->results[p] == DROP);
        }
        batch->has_results = true;

        break; }
    case WRITE_USER_POSTPROC:

        postproc_batch(batch, host_out_ptr);

    case WRITE_FIXED_SEGMENTS: {

        /* Run postporcessing only. */
        for (unsigned p = 0; p < batch->count; p++) {
            if (batch->excluded[p]) continue;
            unsigned elemsz = t->aligned_item_sizes.size;
            unsigned offset = elemsz * p;
            Packet *pkt = Packet::from_base(batch->packets[p]);
            batch->results[p] = elem->postproc(input_port,
                                               (char*) host_out_ptr + offset,
                                               pkt);
            batch->excluded[p] = (batch->results[p] == DROP);
        }
        batch->has_results = true;

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
