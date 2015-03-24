#ifndef __NBA_ELEMENT_STANDARD_CLASSIFIER_HH__
#define __NBA_ELEMENT_STANDARD_CLASSIFIER_HH__


#include <rte_config.h>
#include <rte_memory.h>
#include <rte_mbuf.h>
#include <rte_ether.h>

#include "../../lib/element.hh"
#include "../../lib/annotation.hh"
#include <vector>
#include <queue>
#include <string>

namespace nba {

struct MatchCondition {
    int pos;
    int value;
    struct MatchCondition *next;
};

class Classifier : public Element {
public:
	Classifier(): Element()
	{
	}

	~Classifier()
	{
        std::vector<struct MatchCondition *>::iterator iter = condition_vector.begin();
        struct MatchCondition *condition = nullptr;
        struct MatchCondition *next_condition = nullptr;
        struct MatchCondition *prev_condition = nullptr;
        while (iter != condition_vector.end()) {
            condition = (struct MatchCondition *) *iter;
            next_condition = condition->next;
            while (next_condition != nullptr) {
                prev_condition = next_condition;
                next_condition = next_condition->next;
                free(prev_condition);
            } 
            free(condition);
        }
	}

	const char *class_name() const { return "Classifier"; };
	const char *port_count() const { return "1/*"; };

	int initialize();
	int initialize_global() { return 0; };		// per-system configuration
	int initialize_per_node() { return 0; };	// per-node configuration
	int configure(comp_thread_context *ctx, std::vector<std::string> &args);

	int process(int input_port, struct rte_mbuf *pkt, struct annotation_set *anno);

    std::vector<struct MatchCondition*> condition_vector; 
};

EXPORT_ELEMENT(Classifier);

}

#endif
