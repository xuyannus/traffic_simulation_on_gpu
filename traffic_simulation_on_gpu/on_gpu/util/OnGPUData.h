/*
 * OnGPUData.h
 *
 *  Created on: Feb 9, 2014
 *      Author: xuyan
 */

#ifndef ONGPUDATA_H_
#define ONGPUDATA_H_

#include "../util/shared_gpu_include.h"

class GPUSharedParameter {
public:

	//for small network
	int ON_GPU_LANE_SIZE;
	int ON_GPU_NODE_SIZE;

	int ON_GPU_START_TIME_STEPS;
	int ON_GPU_END_TIME_STEPS;
	int ON_GPU_UNIT_TIME_STEPS; //sec
	int ON_GPU_TOTAL_TIME_STEPS;

	int ON_GPU_MAX_LANE_DOWNSTREAM;
	int ON_GPU_MAX_LANE_UPSTREAM;

	//Length Related
	int ON_GPU_ROAD_LENGTH; //meter
	int ON_GPU_VEHICLE_LENGTH; //meter
	int ON_GPU_MAX_VEHICLE_PER_LANE;
	int ON_GPU_VEHICLE_MAX_LOADING_ONE_TIME;

	//Speed Related
	float ON_GPU_Alpha;
	float ON_GPU_Beta;
	float ON_GPU_Max_Density; //vehicle on road
	float ON_GPU_Min_Density; //vehicle on road
	int ON_GPU_MAX_SPEED;
	int ON_GPU_MIN_SPEED;

	int ON_GPU_LANE_INPUT_CAPACITY_TIME_STEP;
	int ON_GPU_LANE_OUTPUT_CAPACITY_TIME_STEP;

	int ON_GPU_MAX_ROUTE_LENGTH;
	int ON_GPU_MAXIMUM_LANE_CODING_LENGTH;
	int ON_GPU_GPU_TO_CPU_SIMULATION_RESULTS_COPY_BUFFER_SIZE;

};

#endif /* ONGPUDATA_H_ */
