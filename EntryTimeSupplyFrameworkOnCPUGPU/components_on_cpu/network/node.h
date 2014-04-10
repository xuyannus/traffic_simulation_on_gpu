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
	int x;
	int y;
public:
	std::vector<Link*> upstream_links;
	std::vector<Link*> downstream_links;
};

#endif /* NODE_H_ */
