/*
 * Node.h
 *
 *  Created on: Jan 1, 2014
 *      Author: xuyan
 */

#ifndef NODE_H_
#define NODE_H_

#include "../util/shared_cpu_include.h"

//forward declaration
class Link;

class Node {
public:
	int node_id;
	int up_link_start_index;
	int up_link_end_index;
	int up_lane_start_index;
	int up_lane_end_index;
};

#endif /* NODE_H_ */
