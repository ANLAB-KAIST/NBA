/*
 * graphanalysis.cc
 *
 *  Created on: Sep 12, 2014
 *      Author: leeopop
 */

#include <nba/core/bitmap.hh>
#include <nba/framework/graphanalysis.hh>
#include <nba/framework/datablock.hh>
#include <unordered_set>
#include <vector>
#include <unordered_map>
#include <stack>
#include <cassert>
using std::unordered_set;
using std::stack;
using std::vector;
using L::Bitmap;
namespace nba
{

const vector<vector<GraphMetaData*> > &GraphAnalyzer::get_linear_groups()
{
    return linear_group_set;
}

void GraphMetaData::link(GraphMetaData* child)
{
    this->outEdge.insert(child);
    child->inEdge.insert(this);
}

int GraphMetaData::get_linear_group()
{
    return linear_group;
}

void GraphAnalyzer::analyze(ParseInfo* info)
{
    unordered_set<GraphMetaData*> visited;
    int linear_group_id = 0;
    int total_modules = click_num_module(info);
    int total_roots = click_num_root(info);

    /* This loop groups elements into a set of disjoint linear paths. */
    for (int root_id = 0; root_id < total_roots; root_id++) {
        /* Begin with all root elements (if there are disjoint sets of elements). */
        GraphMetaData * root = (GraphMetaData*) click_get_root(info, root_id);
        stack<GraphMetaData*> dfsStack;
        dfsStack.push(root);

        while (!dfsStack.empty()) {
            GraphMetaData * current = dfsStack.top();
            dfsStack.pop();
            visited.insert(current);

            for (GraphMetaData* child : current->outEdge) {
                if (visited.find(child) == visited.end())
                    dfsStack.push(child);
            }

            if (current->inEdge.size() == 1) {
                GraphMetaData* parent = 0;
                for (GraphMetaData* temp : current->inEdge)
                    parent = temp;
                assert(parent);
                if (parent->outEdge.size() == 1)
                    current->linear_group = parent->linear_group;
            }

            if (current->outEdge.size() == 1) {
                if (current->linear_group == -1) {
                    current->linear_group = linear_group_id++;
                    linear_group_set.push_back(vector<GraphMetaData*>());
                    assert((int)linear_group_set.size() == linear_group_id);
                }
            }
            if (current->linear_group != -1)
                linear_group_set[current->linear_group].push_back(current);
        }
    }

    /* This loop detects read-write dependencies along each linear group,
     * using ROI bitmaps. (Currently unused...) */
    #if 0
    for (size_t group_id = 0; group_id < linear_group_set.size(); group_id++) {
        vector<GraphMetaData*> linear_group = linear_group_set[group_id];
        size_t before = 0;
        size_t next = 1;
        Bitmap write_bitmap(2048); //corruption map
        while (next < linear_group.size()) {
            GraphMetaData* before_data = linear_group[before];
            GraphMetaData* next_data = linear_group[next];

            for (size_t dbIndex=0; dbIndex < before_data->dbIndex.size(); dbIndex++) {
                write_bitmap.merge(before_data->dbWriteBitmap[dbIndex]);
                for (size_t dbIndex2=0; dbIndex2 < next_data->dbIndex.size(); dbIndex2++) {
                    Bitmap read_bitmap = before_data->dbReadBitmap[dbIndex];
                    if (!write_bitmap.isCollide(read_bitmap)) {
                        // printf("Datablock %s -> %s do not collide.\n", datablock_names[dbIndex], datablock_names[dbIndex2]);
                    } else {
                        // printf("Datablock %s -> %s collide.\n", datablock_names[dbIndex], datablock_names[dbIndex2]);
                        Bitmap collision(2048);
                        collision.clear();
                        collision.merge(read_bitmap);
                        collision.intersect(write_bitmap);
                        collision.print();
                    }
                }
            }
            before++;
            next++;
        }
    }
    #endif
}

void GraphMetaData::add_roi(int dbIndex, const L::Bitmap& read, const L::Bitmap& write)
{
    this->dbIndex.push_back(dbIndex);
    this->dbReadBitmap.push_back(read);
    this->dbWriteBitmap.push_back(write);
}

}

/* vim: set ts=8 sts=4 sw=4 et: */
