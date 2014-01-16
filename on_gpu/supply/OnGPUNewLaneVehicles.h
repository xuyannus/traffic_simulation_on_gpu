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
	int *lane_ID;
	int *new_vehicle_size;
	GPUVehicle **new_vehicles;

	public NewLaneVehicles(){
		this->lane_ID = (int*)malloc(sizeof(int)*LANE_SIZE);
		this->new_vehicle_size = (int*)malloc(sizeof(int)*LANE_SIZE);
		this->new_vehicles = (int*)malloc(sizeof(int)*LANE_SIZE);
		for(int i=0; i<LANE_SIZE; i++){
			this->new_vehicles[i]=(GPUVehicle*)malloc(sizeof(GPUVehicle)*VEHICLE_MAX_LOADING_ONE_TIME);
		}
		/*
			or init in this way
			GPUVehicle *v = (GPUVehicles*)malloc(sizeof(GPUVehicle) * LANE_SIZE * VEHICLE_MAX_LOADING_ONE_TIME);
			for(int i=0; i<LANE_SIZE; i++)
				this->new_vehicles[i] = v[i*VEHICLE_MAX_LOADING_ONE_TIME];
		*/
	}
};

#endif /* ONGPUNEWSEGMENTVEHICLES_H_ */
