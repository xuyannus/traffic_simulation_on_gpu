/*
 * Link.h
 *
 *  Created on: Jan 1, 2014
 *      Author: xuyan
 */

#ifndef LINK_H_
#define LINK_H_

#include "../util/shared_cpu_include.h"
#include "Node.h"

using namespace std;

class Link {

public:
	int link_id;
	int length;

	Node* from_node;
	Node* to_node;
};

#endif /* LINK_H_ */
