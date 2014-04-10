/*
 * OnGPUNodePool.h
 *
 *  Created on: Jan 2, 2014
 *      Author: xuyan
 */

#ifndef ONGPUNODEPOOL_H_
#define ONGPUNODEPOOL_H_

#include "../../components_on_cpu/util/configurations_on_cpu.h"

class NodePool {
public:
	int node_ID[kNodeSize];
	int MAXIMUM_ACCUMULATED_FLOW[kNodeSize];
	int ACCUMULATYED_UPSTREAM_CAPACITY[kNodeSize];
	int ACCUMULATYED_DOWNSTREAM_CAPACITY[kNodeSize];

	//if -1, means no such lane
	int upstream[kMaxLaneUpstream][kNodeSize];
	int downstream[kMaxLaneDownstream][kNodeSize];
};

#endif /* ONGPUNODEPOOL_H_ */
