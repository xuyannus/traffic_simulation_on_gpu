/*
 * OnGPUNewSegmentVehicles.h
 *
 *  Created on: Dec 31, 2013
 *      Author: xuyan
 */

#ifndef ONGPUNEWLANEVEHICLES_H_
#define ONGPUNEWLANEVEHICLES_H_

#include "on_GPU_vehicle.h"
#include "../../components_on_cpu/util/configurations_on_cpu.h"

class NewLaneVehicles {

public:
	int lane_ID[kLaneSize];
	int new_vehicle_size[kLaneSize];
	int new_vehicles[kLaneSize][kLaneInputCapacityPerTimeStep];

};

#endif /* ONGPUNEWSEGMENTVEHICLES_H_ */
