/*
 * OnGPUNewSegmentVehicles.h
 *
 *  Created on: Dec 31, 2013
 *      Author: xuyan
 */

#ifndef ONGPUNEWLANEVEHICLES_H_
#define ONGPUNEWLANEVEHICLES_H_

#include "OnGPUVehicle.h"
#include "../../main/shared_data.h"

class NewLaneVehicles {

public:
	int lane_ID[LANE_SIZE];
	int new_vehicle_size[LANE_SIZE];
	GPUVehicle new_vehicles[VEHICLE_MAX_LOADING_ONE_TIME][LANE_SIZE];
};

#endif /* ONGPUNEWSEGMENTVEHICLES_H_ */
