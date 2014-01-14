/*
 * OnGPUNodePool.h
 *
 *  Created on: Jan 2, 2014
 *      Author: xuyan
 */

#ifndef ONGPUNODEPOOL_H_
#define ONGPUNODEPOOL_H_

#include "../../main/shared_data.h"

class NodePool {
public:
	int node_ID[NODE_SIZE];
	int MAXIMUM_ACCUMULATED_FLOW[NODE_SIZE];
	int ACCUMULATYED_UPSTREAM_CAPACITY[NODE_SIZE];
	int ACCUMULATYED_DOWNSTREAM_CAPACITY[NODE_SIZE];

	//if -1, means no such lane
	int upstream[MAX_LANE_UPSTREAM][NODE_SIZE];
	int downstream[MAX_LANE_DOWNSTREAM][NODE_SIZE];

};

#endif /* ONGPUNODEPOOL_H_ */
