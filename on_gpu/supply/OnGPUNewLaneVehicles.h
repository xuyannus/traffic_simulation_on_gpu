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
	GPUVehicle **new_vehicles;
	GPUVehicle *v;

public:
	NewLaneVehicles(){
		this->new_vehicles = (GPUVehicle**)malloc(sizeof(GPUVehicle*)*LANE_SIZE);
		this->v = (GPUVehicle*)malloc(sizeof(GPUVehicle)*LANE_SIZE*VEHICLE_MAX_LOADING_ONE_TIME);
		for (int i=0; i<LANE_SIZE; i++) {
			this->new_vehicles[i]=&v[i*VEHICLE_MAX_LOADING_ONE_TIME];
		}
		printf("NewLaneVehicles: size: %d\n", sizeof(GPUVehicle));
	}
};

#endif /* ONGPUNEWSEGMENTVEHICLES_H_ */
