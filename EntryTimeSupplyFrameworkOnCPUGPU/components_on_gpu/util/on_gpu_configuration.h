/*
 * OnGPUCONFIGURATION.h
 *
 *  Created on: Feb 9, 2014
 *      Author: xuyan
 */

#ifndef ONGPUCONFIGURATION_H_
#define ONGPUCONFIGURATION_H_

#include "../util/shared_gpu_include.h"

class GPUSharedParameter {
public:

	//for small network
	int kOnGPULaneSize;
	int kOnGPUNodeSize;

	int kOnGPUStartTimeStep;
	int kOnGPUEndTimeStep;
	int kOnGPUUnitTimeStep; //sec
	int kOnGPUTotalTimeSteps;

	int kOnGPUMaxlaneDownstream;
	int kOnGPUMaxLaneUpstream;

	//Length Related
	int kOnGPURoadLength; //meter
	int kOnGPUVehicleLength; //meter
	int kOnGPUMaxVehiclePerLane;
	int kOnGPUVehicleMaxLoadingOneTime;

	//Speed Related
	float kOnGPUAlpha;
	float kOnGPUBeta;
	float kOnGPUMaxDensity; //vehicle on road
	float kOnGPUMinDensity; //vehicle on road
	int kOnGPUMaxSpeed;
	int kOnGPUMinSpeed;

	int kOnGPULaneInputCapacityPerTimeStep;
	int kOnGPULaneOutputCapacityPerTimeStep;

	int kOnGPUMaxRouteLength;
	int kOnGPUMaxLaneCodingLength;
	int kOnGPUGPUToCPUSimulationResultsCopyBufferSize;

};

#endif /* ONGPUCONFIGURATION_H_ */
