/*
 * graphanalysis.hh
 *
 *  Created on: Sep 12, 2014
 *      Author: leeopop
 */

#ifndef GRAPHANALYSIS_HH_
#define GRAPHANALYSIS_HH_

extern "C" {
#include <click_parser.h>
}
#include <cstdio>
#include <unordered_set>
#include <vector>
#include <nba/core/bitmap.hh>

namespace nba
{

class GraphMetaData;

class GraphAnalyzer
{
public:
    GraphAnalyzer() : analyzed(false) {}
    virtual ~GraphAnalyzer() {}

    /** From the parsing results, it attaches graph analysis results into
     *  individual elements that are already instantiated. */
    void analyze(ParseInfo* pi);

    const std::vector<std::vector<GraphMetaData*> > &get_linear_groups();

private:
    bool analyzed;
    std::vector<std::vector<GraphMetaData*> > linear_group_set;
};

class GraphMetaData
{
private:
    int linear_group;
    std::unordered_set<GraphMetaData*> inEdge;
    std::unordered_set<GraphMetaData*> outEdge;
    std::vector<int> dbIndex;
    std::vector<L::Bitmap> dbReadBitmap;
    std::vector<L::Bitmap> dbWriteBitmap;
public:
    GraphMetaData() : linear_group(-1) { }
    virtual ~GraphMetaData() { }

    void link(GraphMetaData* child);
    int get_linear_group();
    void add_roi(int dbIndex, const L::Bitmap& read, const L::Bitmap& write);

    friend class GraphAnalyzer;
};

}
#endif /* GRAPHANALYSIS_HH_ */

// vim: ts=8 sts=4 sw=4 et
