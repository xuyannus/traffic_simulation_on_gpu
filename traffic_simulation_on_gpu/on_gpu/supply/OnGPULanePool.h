#ifndef LANES_H_
#define LANES_H_

#include "../../main/shared_data.h"
#include "OnGPUVehicle.h"

class LanePool {
public:
	//network
	int lane_ID[LANE_SIZE];
	int from_node_id[LANE_SIZE];
	int to_node_id[LANE_SIZE];

	//ETS framework
	int Tp[LANE_SIZE];
	int Tq[LANE_SIZE];
	float accumulated_offset[LANE_SIZE];

	//measurement
	float flow[LANE_SIZE];
	float density[LANE_SIZE];
	float speed[LANE_SIZE];
	float queue_length[LANE_SIZE];

	/*
	 * for density calculation
	 */
	float lane_length[LANE_SIZE];
	int max_vehicles[LANE_SIZE];
	int output_capacity[LANE_SIZE];
	int input_capacity[LANE_SIZE];
	float empty_space[LANE_SIZE];

	int vehicle_counts[LANE_SIZE];

	/*
	 * for speed calculation
	 */
	float alpha[LANE_SIZE];
	float beta[LANE_SIZE];
	float max_density[LANE_SIZE];
	float min_density[LANE_SIZE];
	float MAX_SPEED[LANE_SIZE];
	float MIN_SPEED[LANE_SIZE];

	/*
	 * for access vehicles
	 */

	GPUVehicle* vehicle_space[MAX_VEHICLE_PER_LANE][LANE_SIZE];

	int vehicle_passed_to_the_lane_counts[LANE_SIZE];
	GPUVehicle* vehicle_passed_space[LANE_INPUT_CAPACITY_TIME_STEP][LANE_SIZE];

	/*
	 * For accumulated length estimation
	 */
	float speed_history[TOTAL_TIME_STEPS][LANE_SIZE];

	/*
	 * For queue length prediction
	 */
	float last_time_empty_space[LANE_SIZE];

	float his_queue_length[QUEUE_LENGTH_HISTORY][LANE_SIZE];
	float his_queue_length_weighting[QUEUE_LENGTH_HISTORY][LANE_SIZE];

	float predicted_queue_length[LANE_SIZE];
	float predicted_empty_space[LANE_SIZE];

	/*
	 * For empty space update
	 */
	int new_vehicle_join_counts[LANE_SIZE];

	/*
	 * Temp Variables
	 */
	bool blocked[LANE_SIZE];

	/**
	 * for debug
	 */
	float debug_data[LANE_SIZE];

};

#endif /* LANES_H_ */
