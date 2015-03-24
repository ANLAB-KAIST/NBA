/*
 * graphanalysis.hh
 *
 *  Created on: Sep 12, 2014
 *      Author: leeopop
 */

#ifndef GRAPHANALYSIS_HH_
#define GRAPHANALYSIS_HH_

extern "C"
{
#include <stdio.h>
#include <click_parser.h>
}

#include <unordered_set>
#include <vector>
#include "bitmap.hpp"

namespace nba
{

class GraphAnalysis
{
private:
	GraphAnalysis();
	~GraphAnalysis();
public:
	static void analyze(ParseInfo* pi);
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
	GraphMetaData();
	virtual ~GraphMetaData();

	void link(GraphMetaData* child);
	int getLinearGroup();
	void addROI(int dbIndex, const L::Bitmap& read, const L::Bitmap& write);

	friend class GraphAnalysis;
};

}
#endif /* GRAPHANALYSIS_HH_ */
