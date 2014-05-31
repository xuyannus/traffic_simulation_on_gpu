/*
 * Link.h
 *
 *  Created on: Jan 1, 2014
 *      Author: xuyan
 */

#ifndef LINK_H_
#define LINK_H_

#include "../util/shared_cpu_include.h"
#include "node.h"

class Link {
public:
	int link_id;
	int link_connection_start;
	int link_connection_end;
	int vehicle_start;
	int vehicle_end;
	int buffered_vehicle_start;
	int buffered_vehicle_end;

	float length; //unit: meters
};

#endif /* LINK_H_ */
