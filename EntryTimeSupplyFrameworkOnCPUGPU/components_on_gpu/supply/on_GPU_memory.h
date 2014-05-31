/*
 * OnGPUMemory.h
 *
 *  Created on: Jan 2, 2014
 *      Author: xuyan
 */

#ifndef ONGPUMEMORY_H_
#define ONGPUMEMORY_H_

#include "../util/shared_gpu_include.h"

#include "on_GPU_lane_pool.h"
#include "on_GPU_new_lane_vehicles.h"
#include "on_GPU_node_pool.h"
#include "on_GPU_lane_vehicle_pool.h"
#include "on_GPU_lane_buffered_vehicle_pool.h"

class GPUMemory {
public:

	LanePool lane_pool;
	NodePool node_pool;

	LaneVehiclePool lane_vehicle_pool;
	LaneBufferedVehiclePool lane_buffered_vehicle_pool;

	NewLaneVehicles new_vehicles_every_time_step[kTotalTimeSteps];

public:

	size_t total_size() {
		return sizeof(LanePool) + sizeof(NodePool) + sizeof(NewLaneVehicles) * kTotalTimeSteps;
	}
};

#endif /* ONGPUMEMORY_H_ */
