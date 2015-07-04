#ifndef __NBA_ELEMGRAPH_HH__
#define __NBA_ELEMGRAPH_HH__

#include "element.hh"
#include "types.hh"
#include "queue.hh"
#include <vector>

#include "computation.hh"

namespace nba {

#define ROOT_ELEMENT (nullptr)

class Element;

class ElementGraph {
public:
    ElementGraph(comp_thread_context *ctx);
    virtual ~ElementGraph() {}

    int count()
    {
        return elements.size();
    }

    /* Executes the element graph for the given batch and free it after
     * processing.  Internally it manages a queue to handle diverged paths
     * with multiple batches to multipe outputs.
     * When it needs to stop processing and wait for asynchronous events
     * (e.g., completion of offloading or release of resources), it moves
     * the batch to the delayed_batches queue. */
    void run(PacketBatch *batch, Element *start_elem, int input_port = 0);

    /* Tries to execute all pending offloaded tasks.
     * This method does not allocate/free any batches. */
    void flush_offloaded_tasks();

    /* Tries to run all delayed batches. */
    void flush_delayed_batches();

    /**
     * A special case on completion of offloading.
     * It begins DFS-based element graph traversing from the given
     * offloaded element, with all results already calculated in the
     * coprocessor thread.
     */
    void enqueue_postproc_batch(PacketBatch *batch, Element *offloaded_elem,
                                int input_port);

    /**
     * Check if the given datablock (represented as a global ID) is reused
     * after the given offloaded element in a same (linear?) path.
     */
    bool check_datablock_reuse(Element *offloaded_elem, int datablock_id);
    bool check_next_offloadable(Element *offloaded_elem);

    /**
     * Add a new element instance to the graph.
     * If prev_elem is NULL, it becomes the root of a subtree.
     */
    int add_element(Element *new_elem);
    int link_element(Element *to_elem, int input_port,
                     Element *from_elem, int output_port);

    SchedulableElement *get_entry_point(int entry_point_idx = 0);


    /**
     * Validate the element graph.
     * Currently, this checks the writer-reader pairs of structured
     * annotations.
     */
    int validate();

    /**
     * Returns the list of schedulable elements.
     * They are executed once on every polling iteration.
     */
    const FixedRing<SchedulableElement*, nullptr>& get_schedulable_elements() const;

    /**
     * Returns the list of all elements.
     */
    const FixedRing<Element*, nullptr>& get_elements() const;

    /**
     * Free a packet batch.
     */
    void free_batch(PacketBatch *batch, bool free_pkts = true);

    /* TODO: calculate from the actual graph */
    static const int num_max_outputs = 16;
protected:
    /**
     * Used to book-keep element objects.
     */
    FixedRing<Element*, nullptr> elements;
    FixedRing<SchedulableElement*, nullptr> sched_elements;

    /**
     * Used to pass context objects when calling element handlers.
     */
    comp_thread_context *ctx;

    FixedRing<PacketBatch *, nullptr> queue;
    FixedRing<OffloadTask *, nullptr> ready_tasks[NBA_MAX_PROCESSOR_TYPES];
    FixedRing<PacketBatch *, nullptr> delayed_batches;

private:
    SchedulableElement *input_elem;

};

}

#endif

// vim: ts=8 sts=4 sw=4 et
