#ifndef __NBA_ELEMGRAPH_HH__
#define __NBA_ELEMGRAPH_HH__

#include <nba/core/queue.hh>
#include <nba/framework/computation.hh>
#include <nba/framework/threadcontext.hh>
#include <nba/framework/task.hh>
#include <nba/element/element.hh>
#include <nba/element/packetbatch.hh>
#include <vector>
#include <map>

struct rte_hash;

namespace nba {

#define ROOT_ELEMENT (nullptr)

enum ElementOffloadingActions : int {
    ELEM_OFFL_NOTHING = 0,
    ELEM_OFFL_PREPROC = 1,
    ELEM_OFFL_POSTPROC = 2,
    ELEM_OFFL_POSTPROC_FIN = 4,
};

class Element;
class OffloadTask;
class PacketBatch;

struct offload_action_key {
    void *elemptr;
    int dbid;
    int action;
};

class ElementGraph {
public:
    ElementGraph(comp_thread_context *ctx);
    virtual ~ElementGraph() {}

    int count()
    {
        return elements.size();
    }

    /* Inserts the given batch/offloadtask to the internal task queue.
     * This does not execute the pipeline; call flush_tasks() for that. */
    void enqueue_batch(PacketBatch *batch, Element *start_elem, int input_port = 0);
    void enqueue_offload_task(OffloadTask *otask, Element *start_elem, int input_port = 0);

    /* Tries to run all pending computation tasks. */
    void flush_tasks();

    /* Scans and executes dispatch() handlers of schedulable elements.
     * This implies scan_offloadable_elements() since offloadable elements
     * inherits schedulable elements. */
    void scan_schedulable_elements(uint64_t loop_count);

    /* Scans and executes dispatch() handlers of offloadable elements.
     * It is a shorthand version that ignores next_delay output arguments.
     * This fetches the GPU-processed batches and feed them into the graph
     * again. */
    void scan_offloadable_elements(uint64_t loop_count);

    /* Start processing with the given batch and the entry point. */
    void feed_input(int entry_point_idx, PacketBatch *batch, uint64_t loop_count);

    void add_offload_action(struct offload_action_key *key);
    bool check_preproc(OffloadableElement *oel, int dbid);
    bool check_postproc(OffloadableElement *oel, int dbid);
    bool check_postproc_all(OffloadableElement *oel);

    bool check_next_offloadable(Element *offloaded_elem);
    Element *get_first_next(Element *elem);

    /**
     * Add a new element instance to the graph.
     * If prev_elem is NULL, it becomes the root of a subtree.
     */
    int add_element(Element *new_elem);
    int link_element(Element *to_elem, int input_port,
                     Element *from_elem, int output_port);

    /**
     * Validate the element graph.
     * Currently, this checks the writer-reader pairs of structured
     * annotations.
     */
    int validate();

    /**
     * Returns the list of all elements.
     */
    const FixedRing<Element*, nullptr>& get_elements() const;

    /**
     * Free a packet batch.
     */
    void free_batch(PacketBatch *batch, bool free_pkts = true);

    /* TODO: calculate from the actual graph */
    static const int num_max_outputs = NBA_MAX_ELEM_NEXTS;

private:
    /**
     * Used to book-keep element objects.
     */
    FixedRing<Element *, nullptr> elements;

    /**
     * Book-keepers to avoid dynamic_cast and runtime type checks.
     */
    FixedRing<SchedulableElement *, nullptr> sched_elements;
    FixedRing<OffloadableElement *, nullptr> offl_elements;

    /**
     * Used to pass context objects when calling element handlers.
     */
    comp_thread_context *ctx;

    FixedRing<void *, nullptr> queue;

    /* Executes the element graph for the given batch and free it after
     * processing.  Internally it manages a queue to handle diverged paths
     * with multiple batches to multipe outputs.
     * When it needs to stop processing and wait for asynchronous events
     * (e.g., completion of offloading or release of resources), it moves
     * the batch to the delayed_batches queue. */
    void process_batch(PacketBatch *batch);
    void process_offload_task(OffloadTask *otask);
    void send_offload_task_to_device(OffloadTask *task);

    struct rte_hash *offl_actions;

    /* The entry point of packet processing pipeline (graph). */
    SchedulableElement *input_elem;
};

}

#endif

// vim: ts=8 sts=4 sw=4 et
