#ifndef GPUVEHICLE_H_
#define GPUVEHICLE_H_

#include "../../main/shared_data.h"

//const int VEHICLE_LENGTH = 5; //meter
//const int VEHICLE_MAX_SIZE = 220 * 1000 / 5; //each lane contains 200 vehicles in maxminum

class GPUVehicle {
public:

	int vehicle_ID;
	int current_lane_ID;
	int entry_time;

	int whole_path_length;
	int next_path_index;
	int path_code[MAXIMUM_LANE_CODING_LENGTH];
};

#endif /* GPUVEHICLE_H_ */
