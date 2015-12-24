#ifndef __NBA_DATABLOCK_HH__
#define __NBA_DATABLOCK_HH__

#include <nba/framework/config.hh>
#include <nba/framework/computecontext.hh>
#include <vector>
#include <string>
#include <tuple>

namespace nba {

/* Forward declarations. */
class OffloadableElement;
class PacketBatch;
class DataBlock;

typedef DataBlock* (*datablock_constructor)(void);

extern size_t num_datablocks;
extern const char* datablock_names[NBA_MAX_DATABLOCKS];
extern datablock_constructor datablock_ctors[NBA_MAX_DATABLOCKS];

void declare_datablock_impl(const char *name, datablock_constructor ctor, int &your_id);
#define __combine_direct(x,y) x ## y
#define __combine(x,y) __combine_direct(x,y)
#define __make_uniq(category,id) __combine(__combine(category, id), __COUNTER__)

/* Use the following macro to define a data block.
 * If the name of data block overlaps, it will only have a single same
 * index and the data block can be consistently shared across different
 * modules. */

#define declare_datablock(name,ctor,id) \
void __make_uniq(_declare_datablock_, id)(void);\
void __attribute__((constructor, used)) __make_uniq(_declare_datablock_, id)(void) { \
    declare_datablock_impl(name, ctor, id); \
}

/* ROI(region of interest) data structures for offloading */
enum ReadROIType {
    READ_NONE           = 0,
    READ_PARTIAL_PACKET = 1,  // Packet segment with fixed size.
    READ_WHOLE_PACKET   = 2,  // Whole packet. (whose size can be dynamic)
    READ_USER_PREPROC   = 3,  // Uses preproc_batch() instead of direct copy.
};

enum WriteROIType {
    WRITE_NONE           = 0,
    WRITE_PARTIAL_PACKET = 1,  // (same as ReadROIType)
    WRITE_WHOLE_PACKET   = 2,  // (same as ReadROIType)
    WRITE_FIXED_SEGMENTS = 3,  // An array of items having a same fixed size
    WRITE_USER_POSTPROC  = 4,  // Uses postproc_batch() instead of direct copy.
    /* In all cases, el->postproc_packet() is called always to set batch->results. */
};

struct read_roi_info {
    enum ReadROIType type;
    size_t offset;
    int length;
    int align;

    /* FIXME: packet size가 변하는 경우 추가할 바이트 수.
     *        현재는 SHA_DIGEST_LENGTH만큼 미리 잡아서 복사해놓는 목적으로 사용.
     *        ROI joining 구현하면 자동화 가능할듯? */
    int size_delta;
};

struct write_roi_info {
    enum WriteROIType type;
    size_t offset;
    int length;
    int align;
};

#ifdef NBA_NO_HUGE
/* WARNING: you should use minimum packet sizes for IPsec. */
struct item_size_info {
    union {
        uint16_t size;
        uint16_t sizes[NBA_MAX_COMP_BATCH_SIZE * 12];
    };
    uint16_t offsets[NBA_MAX_COMP_BATCH_SIZE * 12];
};
#else
struct item_size_info {
    union {
        uint16_t size;
        uint16_t sizes[NBA_MAX_COMP_BATCH_SIZE * 96];
    };
    uint16_t offsets[NBA_MAX_COMP_BATCH_SIZE * 96];
};
#endif

/** Datablock tracking struct.
 *
 * It resides in PacketBatch as a static array, and keeps track of the
 * status of data blocks attached to the batch.
 */
struct datablock_tracker {
    void *host_in_ptr;
    memory_t dev_in_ptr;
    void *host_out_ptr;
    memory_t dev_out_ptr;
    size_t in_size;
    size_t in_count;
    size_t out_size;
    size_t out_count;
    //struct item_size_info exact_item_sizes;
    struct item_size_info *aligned_item_sizes_h;
    memory_t aligned_item_sizes_d;
};

/* NOTE: The alignment of this struct should match with CUDA. */
struct datablock_kernel_arg {
    uint32_t total_item_count_in;
    uint32_t total_item_count_out;
    void *buffer_bases_in[NBA_MAX_COPROC_PPDEPTH];
    void *buffer_bases_out[NBA_MAX_COPROC_PPDEPTH];
    uint32_t item_count_in[NBA_MAX_COPROC_PPDEPTH];
    uint32_t item_count_out[NBA_MAX_COPROC_PPDEPTH];
    union {
        uint16_t item_size_in;
        uint16_t *item_sizes_in[NBA_MAX_COPROC_PPDEPTH];
    };
    union {
        uint16_t item_size_out;
        uint16_t *item_sizes_out[NBA_MAX_COPROC_PPDEPTH];
    };
    uint16_t *item_offsets_in[NBA_MAX_COPROC_PPDEPTH];
    uint16_t *item_offsets_out[NBA_MAX_COPROC_PPDEPTH];
}; // __attribute__((aligned(8)));


/** Datablock information class.
 *
 * It defines ROI and the operations of a specific datablock.
 * It resides in the computation context and its instances are stateless.
 */
class DataBlock
{
public:
    DataBlock() : merged_datablock_idx(-1) { }

    virtual ~DataBlock() { }

    int merged_datablock_idx;
    size_t merged_read_offset;
    size_t merged_write_offset;

    /* Set when instantiating datablocks in main.cc */
    void set_id(int id) { my_id = id; }

    /* Used when referencing selves. */
    int get_id() const { return my_id; }

    virtual const char *name() const = 0;
    virtual void get_read_roi(struct read_roi_info *roi) const = 0;
    virtual void get_write_roi(struct write_roi_info *roi) const = 0;
    virtual void *get_invalid_value() { return nullptr; }

    /* The return values of following methods are (bytes, count). */
    std::tuple<size_t, size_t> calc_read_buffer_size(PacketBatch *batch);
    std::tuple<size_t, size_t> calc_write_buffer_size(PacketBatch *batch);

    void preprocess(PacketBatch *batch, void *host_ptr);
    void postprocess(OffloadableElement *elem, int input_port, PacketBatch *batch, void *host_ptr);

    /* Below methods arre used only when ROI type is USER_PREPROC/USER_POSTPROC. */
    virtual void calculate_read_buffer_size(PacketBatch *batch, size_t &out_bytes, size_t &out_count)
    { out_bytes = 0; out_count = 0; }
    virtual void calculate_write_buffer_size(PacketBatch *batch, size_t &out_bytes, size_t &out_count)
    { out_bytes = 0; out_count = 0; }
    virtual void preproc_batch(PacketBatch *batch, void *buffer) { }
    virtual void postproc_batch(PacketBatch *batch, void *buffer) { }

private:
    int my_id;

    size_t read_buffer_size;
    size_t write_buffer_size;
};

class MergedDataBlock : DataBlock
{
public:
    MergedDataBlock(void)
    {
        memset(src_datablocks, 0, sizeof(DataBlock *) * NBA_MAX_DATABLOCKS);
        // TODO: allocate(?) dbid
    }

    virtual ~MergedDataBlock() { }

    void add_datablock(DataBlock *db)
    {
        // TODO: implement
        //       merge the offset/length of read/write ROIs
        db->merged_datablock_idx = this->dbid;
        // db->offset = ...;
    }

private:
    int dbid;
    struct read_roi_info *read_roi;
    struct write_roi_info *write_roi;
    DataBlock *src_datablocks[NBA_MAX_DATABLOCKS];
};

}

#endif

// vim: ts=8 sts=4 sw=4 et
