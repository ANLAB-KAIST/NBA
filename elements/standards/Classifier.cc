#include "Classifier.hh"
#include <queue>
#include <rte_memory.h>
#include <rte_ether.h>

using namespace std;
using namespace nba;

int Classifier::initialize()
{
    return 0;
}

int Classifier::configure(comp_thread_context *ctx, std::vector<std::string> &args)
{
    Element::configure(ctx, args);

    // Temporarily store conditions before connecting each other.
    queue<struct MatchCondition *> temp_queue;

    struct MatchCondition *cur_condition = nullptr;

    size_t delim_index, index;
    struct MatchCondition *newCondition, *headCondition, *next_condition;
    vector<std::string>::iterator iter_str;
    for (iter_str = args.begin(); iter_str != args.end(); iter_str++) {
        assert(temp_queue.empty());
        std::string cur_str = (std::string) *iter_str;

        // Init condition of inner while loop.
        delim_index = cur_str.find_first_of(" ");
        index = cur_str.find_first_of("/");
        newCondition = headCondition = next_condition = nullptr;

        bool is_ended = false;
        std::string position_str, byte_str;
        while (!is_ended)
        {
            if (delim_index == string::npos) {
                is_ended = true;
            }

            position_str = cur_str.substr(0, index);
            byte_str = cur_str.substr(index+1, delim_index-index);
            //printf("Position: %s, Byte: %s, delim_index: %lu, index: %lu, is_ended: %d\n", position_str.c_str(), byte_str.c_str(), delim_index, index, is_ended);
            cur_str = cur_str.substr(delim_index+1, string::npos);
            //printf("new cur_str: %s\n", cur_str.c_str());

            newCondition = (struct MatchCondition *) malloc(sizeof(struct MatchCondition));
            newCondition->pos = stoi(position_str, NULL, 10);
            newCondition->value = stoi(byte_str, NULL, 16);
            newCondition->next = nullptr; // Just for now..
            temp_queue.push(newCondition);

            if (headCondition == nullptr) {
                headCondition = newCondition;
                condition_vector.push_back(headCondition);
            }

            delim_index = cur_str.find_first_of(" ");
            index = cur_str.find_first_of("/");
        }

        // Connecting AND-related conditions
        while (!temp_queue.empty()) {
            cur_condition = temp_queue.front();
            temp_queue.pop();

            if (!temp_queue.empty()) {
                cur_condition->next = temp_queue.front();
                assert(cur_condition->next != nullptr);
            }
            else {
                cur_condition->next = nullptr;
                break;
            }
        }
    }

/*
    // Print out to check
    vector<struct MatchCondition *>::iterator iter_cond;
    int rule_cnt;
    for (rule_cnt = 1, iter_cond = condition_vector.begin(); iter_cond != condition_vector.end(); iter_cond++, rule_cnt++) {
        cur_condition = (struct MatchCondition *) *iter_cond;
        printf("Rule %d-1, pos: %d, value: %x\n", rule_cnt, cur_condition->pos, cur_condition->value);

        next_condition = cur_condition->next;
        int cnt = 1;
        while (next_condition != nullptr) {
            printf("Rule %d-%d, pos: %d, value: %x\n", rule_cnt, cnt, next_condition->pos, next_condition->value);
            next_condition = next_condition->next;
            cnt++;
        }
    }
*/
    return 0;
}

int Classifier::process(int input_port, Packet *pkt)
{
    char *packet = (char *) pkt->data();
    int output_port = 0;

    vector<struct MatchCondition *>::iterator iter2;
    struct MatchCondition *cur_condition = nullptr;
    short *p_value_to_compare = nullptr;
    bool is_match;
    unsigned short value, value2;
    for (iter2 = condition_vector.begin(); iter2 != condition_vector.end(); iter2++) {
        is_match = true;
        cur_condition = (struct MatchCondition *) *iter2;
        while (true) {
            value = packet[cur_condition->pos];
            value2 = packet[cur_condition->pos + 1];
            //printf("Value: %x, %x\n", value, value2);
            value = ((value & 0x00FF) << 8) + (value2 & 0x00FF);
            //p_value_to_compare = static_cast<int*>(&packet[cur_condition->pos + 8]);
            //p_value_to_compare = reinterpret_cast<short*>(ptr);

            //if (cur_condition->value != *p_value_to_compare) {
            if (cur_condition->value != value) {
                //printf("cur_condition->value: %x, input packet value: %x\n", cur_condition->value, value);
                is_match = false;
                break;
            }
            if (cur_condition->next == nullptr) {
                break;
            }
            cur_condition = cur_condition->next;
        }
        if (is_match) {
            output(output_port).push(pkt);
            return 0;
        }
        output_port++;
    }

    pkt->kill();
    return 0;
}

// vim: ts=8 sts=4 sw=4 et
