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

GraphMetaData::GraphMetaData()
{
    this->linear_group = -1;
}

GraphMetaData::~GraphMetaData()
{

}

void GraphMetaData::link(GraphMetaData* child)
{
    this->outEdge.insert(child);
    child->inEdge.insert(this);
}

int GraphMetaData::getLinearGroup()
{
    return linear_group;
}

void GraphAnalysis::analyze(ParseInfo* info)
{
    unordered_set<GraphMetaData*> visited;
    int linear_group_id = 0;
    int total_modules = click_num_module(info);
    int total_roots = click_num_root(info);
    vector<vector<GraphMetaData*> > linear_group_set;

    for (int root_id = 0; root_id < total_roots; root_id++)
    {
        //for all root elements
        GraphMetaData * root = (GraphMetaData*)click_get_root(info, root_id);
        stack<GraphMetaData*> dfsStack;
        dfsStack.push(root);

        while (!dfsStack.empty())
        {
            GraphMetaData * current = dfsStack.top();
            dfsStack.pop();
            visited.insert(current);

            for (GraphMetaData* child : current->outEdge)
            {
                if (visited.find(child) == visited.end())
                    dfsStack.push(child);
            }

            if (current->inEdge.size() == 1)
            {
                GraphMetaData* parent = 0;
                for (GraphMetaData* temp : current->inEdge)
                    parent = temp;
                assert(parent);
                if (parent->outEdge.size() == 1)
                    current->linear_group = parent->linear_group;
            }

            if (current->outEdge.size() == 1)
            {
                if (current->linear_group == -1)
                {
                    current->linear_group = linear_group_id++;
                    linear_group_set.push_back(vector<GraphMetaData*>());
                    assert((int)linear_group_set.size() == linear_group_id);
                }
            }
            if (current->linear_group != -1)
                linear_group_set[current->linear_group].push_back(current);
        }
    }

    for (size_t group_id = 0; group_id < linear_group_set.size(); group_id++)
    {
        vector<GraphMetaData*> linear_group = linear_group_set[group_id];
        printf("Traveling linear group_id %lu with size(%lu):\n", group_id, linear_group.size());

        size_t before = 0;
        size_t next = 1;
        Bitmap write_bitmap(2048); //corruption map
        while (next < linear_group.size())
        {
            GraphMetaData* before_data = linear_group[before];
            GraphMetaData* next_data = linear_group[next];

            for (size_t dbIndex = 0; dbIndex < before_data->dbIndex.size(); dbIndex++)
            {
                write_bitmap.merge(before_data->dbWriteBitmap[dbIndex]);
                for (size_t dbIndex2 = 0; dbIndex2 < next_data->dbIndex.size(); dbIndex2++)
                {
                    Bitmap read_bitmap = before_data->dbReadBitmap[dbIndex];

                    if (!write_bitmap.isCollide(read_bitmap))
                    {
                        printf("Datablock %s -> %s do not collide.\n", datablock_names[dbIndex], datablock_names[dbIndex2]);
                    }
                    else
                    {
                        printf("Datablock %s -> %s collide.\n", datablock_names[dbIndex], datablock_names[dbIndex2]);
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
}

void GraphMetaData::addROI(int dbIndex, const L::Bitmap& read, const L::Bitmap& write)
{
    this->dbIndex.push_back(dbIndex);
    this->dbReadBitmap.push_back(read);
    this->dbWriteBitmap.push_back(write);
}

}

/* vim: set ts=8 sts=4 sw=4 et: */
