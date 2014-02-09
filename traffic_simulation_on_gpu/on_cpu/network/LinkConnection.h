/*
 * Link.h
 *
 *  Created on: Jan 1, 2014
 *      Author: xuyan
 */

#ifndef LinkConnection_
#define LinkConnection_

#include "../util/shared_cpu_include.h"

using namespace std;

class LinkConnection {

public:
	int from_link_id;
	int to_link_id;
	bool is_broken;
};

#endif /* LinkConnection_ */
