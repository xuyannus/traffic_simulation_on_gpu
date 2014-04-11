#ifndef LANES_H_
#define LANES_H_

#include "../../components_on_cpu/util/configurations_on_cpu.h"
#include "on_GPU_vehicle.h"

class LanePool {
public:
	//network
	int lane_ID[kLaneSize];
	int from_node_id[kLaneSize];
	int to_node_id[kLaneSize];

	//ETS framework
	int Tp[kLaneSize];
	int Tq[kLaneSize];
	float accumulated_offset[kLaneSize];

	//measurement
	float flow[kLaneSize];
	float density[kLaneSize];
	float speed[kLaneSize];
	float queue_length[kLaneSize];

	//for density calculation
	float lane_length[kLaneSize];
	int max_vehicles[kLaneSize];
	int output_capacity[kLaneSize];
	int input_capacity[kLaneSize];
	float empty_space[kLaneSize];

	int vehicle_counts[kLaneSize];

	//for speed calculation
	float alpha[kLaneSize];
	float beta[kLaneSize];
	float max_density[kLaneSize];
	float min_density[kLaneSize];
	float MAX_speed[kLaneSize];
	float MIN_speed[kLaneSize];

	//for access vehicles
	GPUVehicle* vehicle_space[kMaxVehiclePerLane][kLaneSize];
	int vehicle_passed_to_the_lane_counts[kLaneSize];
	GPUVehicle* vehicle_passed_space[kLaneInputCapacityPerTimeStep][kLaneSize];

	//For accumulated length estimation
	float speed_history[kTotalTimeSteps][kLaneSize];

	//For queue length prediction
	float last_time_empty_space[kLaneSize];
	float his_queue_length[kQueueLengthHistory][kLaneSize];
	float his_queue_length_weighting[kQueueLengthHistory][kLaneSize];

	float predicted_queue_length[kLaneSize];
	float predicted_empty_space[kLaneSize];

	//For empty space update
	int new_vehicle_join_counts[kLaneSize];

	//Temp Variables
	bool blocked[kLaneSize];

	//for debug, not used on GPU
	float debug_data[kLaneSize];

};

#endif /* LANES_H_ */
