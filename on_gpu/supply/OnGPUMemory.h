/*
 * OnGPUMemory.h
 *
 *  Created on: Jan 2, 2014
 *      Author: xuyan
 */

#ifndef ONGPUMEMORY_H_
#define ONGPUMEMORY_H_

#include "../util/shared_gpu_include.h"

#include "OnGPULanePool.h"
#include "OnGPUNewLaneVehicles.h"
#include "OnGPUNodePool.h"

class GPUMemory {
public:

	LanePool lane_pool;
	NodePool node_pool;

	//Vehicles' objects are kept in NewLaneVehicles
	NewLaneVehicles new_vehicles_every_time_step[TOTAL_TIME_STEPS];

//	int test;

public:
	int total_size() {
//		return sizeof(LanePool) + sizeof(NodePool) + sizeof(NewLaneVehicles) * TOTAL_TIME_STEPS + sizeof(int);
		return sizeof(LanePool) + sizeof(NodePool) + sizeof(NewLaneVehicles) * TOTAL_TIME_STEPS;
	}
};

#endif /* ONGPUMEMORY_H_ */
