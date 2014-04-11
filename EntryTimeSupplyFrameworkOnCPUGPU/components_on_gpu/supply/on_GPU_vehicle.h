#ifndef GPUVEHICLE_H_
#define GPUVEHICLE_H_

#include "../../components_on_cpu/util/configurations_on_cpu.h"

class GPUVehicle {
public:
	int vehicle_ID;
	int current_lane_ID;
	int entry_time;

	int whole_path_length;
	int next_path_index;
	int path_code[kMaxLaneCodingLength];
};

#endif /* GPUVEHICLE_H_ */
