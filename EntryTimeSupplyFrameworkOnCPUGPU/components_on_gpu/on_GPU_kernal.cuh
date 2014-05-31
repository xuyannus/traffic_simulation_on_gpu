#ifndef ONGPU_KERNAL_H_
#define ONGPU_KERNAL_H_

#include "../components_on_gpu/supply/on_GPU_memory.h"
#include "../components_on_gpu/supply/on_GPU_vehicle.h"
#include "../components_on_gpu/supply/on_GPU_new_lane_vehicles.h"
#include "../components_on_gpu/util/shared_gpu_include.h"
#include "../components_on_gpu/util/on_gpu_configuration.h"
#include "../components_on_gpu/on_GPU_Macro.h"

#include "../components_on_cpu/util/simulation_results.h"

//Supply Function
__global__ void SupplySimulationPreVehiclePassing(GPUMemory* gpu_data, int time_step, int segment_length, GPUSharedParameter* data_setting_gpu, GPUVehicle *vpool_gpu);
__global__ void SupplySimulationVehiclePassing(GPUMemory* gpu_data, int time_step, int node_length, GPUSharedParameter* data_setting_gpu, GPUVehicle *vpool_gpu);
__global__ void SupplySimulationAfterVehiclePassing(GPUMemory* gpu_data, int time_step, int segment_length, GPUSharedParameter* data_setting_gpu);
__global__ void supply_simulated_results_to_buffer(GPUMemory* gpu_data, int time_step, int segment_length, SimulationResults* buffer, GPUSharedParameter* data_setting_gpu);
__device__ int GetNextVehicleAtNode(GPUMemory* gpu_data, int node_index, int* lane_index, GPUSharedParameter* data_setting_gpu, GPUVehicle *vpool_gpu);

//Utility Function
__device__ float MinOnDevice(float one_value, float the_other);
__device__ float MaxOnDevice(float one_value, float the_other);

/*
 * Supply Function Implementation
 */
__global__ void SupplySimulationPreVehiclePassing(GPUMemory* gpu_data, int time_step, int segment_length, GPUSharedParameter* data_setting_gpu, GPUVehicle *vpool_gpu) {

	int lane_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (lane_index >= segment_length)
		return;

	int time_index = time_step;

//	gpu_data->lane_pool.new_vehicle_join_counts[lane_index] = 0;

//init capacity
	gpu_data->lane_pool.input_capacity[lane_index] = data_setting_gpu->kOnGPULaneInputCapacityPerTimeStep;
	gpu_data->lane_pool.output_capacity[lane_index] = data_setting_gpu->kOnGPULaneOutputCapacityPerTimeStep;

//init for next GPU kernel function
	gpu_data->lane_pool.blocked[lane_index] = false;

//load passed vehicles to the back of the lane
	for (int i = 0; i < gpu_data->lane_pool.vehicle_passed_to_the_lane_counts[lane_index]; i++) {
		if (gpu_data->lane_pool.vehicle_counts[lane_index] < gpu_data->lane_pool.max_vehicles[lane_index]) {
			int new_vehicle_index = gpu_data->lane_pool.vehicle_counts[lane_index] + gpu_data->lane_pool.vehicle_start_index[lane_index];
			int new_buffer_vehicle_index = i + gpu_data->lane_pool.buffered_vehicle_start_index[lane_index];

			//pass the vehicle
			gpu_data->lane_vehicle_pool.vehicle_space[new_vehicle_index] = gpu_data->lane_buffered_vehicle_pool.buffered_vehicle_space[new_buffer_vehicle_index];
			gpu_data->lane_pool.vehicle_counts[lane_index]++;

			//gpu_data->lane_pool.new_vehicle_join_counts[lane_index]++;
		}
	}

	if (gpu_data->lane_pool.vehicle_passed_to_the_lane_counts[lane_index] > 0) {
		gpu_data->lane_pool.empty_space[lane_index] = MinOnDevice(gpu_data->lane_pool.speed[lane_index] * data_setting_gpu->kOnGPUUnitTimeStep, gpu_data->lane_pool.empty_space[lane_index])
				- gpu_data->lane_pool.vehicle_passed_to_the_lane_counts[lane_index] * data_setting_gpu->kOnGPUVehicleLength;

		if (gpu_data->lane_pool.empty_space[lane_index] < 0)
			gpu_data->lane_pool.empty_space[lane_index] = 0;
	}

	gpu_data->lane_pool.last_time_empty_space[lane_index] = gpu_data->lane_pool.empty_space[lane_index];
	gpu_data->lane_pool.vehicle_passed_to_the_lane_counts[lane_index] = 0;

//
//load newly generated vehicles to the back of the lane
	for (int i = 0; i < gpu_data->new_vehicles_every_time_step[time_index].new_vehicle_size[lane_index]; i++) {
		if (gpu_data->lane_pool.vehicle_counts[lane_index] < gpu_data->lane_pool.max_vehicles[lane_index]) {
			int new_vehicle_index = gpu_data->lane_pool.vehicle_counts[lane_index] + gpu_data->lane_pool.vehicle_start_index[lane_index];

			gpu_data->lane_vehicle_pool.vehicle_space[new_vehicle_index] = (gpu_data->new_vehicles_every_time_step[time_index].new_vehicles[lane_index][i]);
			gpu_data->lane_pool.vehicle_counts[lane_index]++;

//			gpu_data->lane_pool.new_vehicle_join_counts[lane_index]++;
		}
	}

	float density_ = 0.0f;
	float speed_ = 0.0f;

//update speed and density
	density_ = 1.0 * data_setting_gpu->kOnGPUVehicleLength * gpu_data->lane_pool.vehicle_counts[lane_index] / gpu_data->lane_pool.lane_length[lane_index];

	if (density_ < data_setting_gpu->kOnGPUMinDensity)
		speed_ = data_setting_gpu->kOnGPUMaxSpeed;
	else {
		speed_ = data_setting_gpu->kOnGPUMaxSpeed
				- data_setting_gpu->kOnGPUMaxSpeed / (data_setting_gpu->kOnGPUMaxDensity - data_setting_gpu->kOnGPUMinDensity) * (density_ - data_setting_gpu->kOnGPUMinDensity);
	}
//		gpu_data->lane_pool.speed[lane_index] = ( gpu_data->lane_pool.MAX_SPEED[lane_index] - gpu_data->lane_pool.MIN_SPEED ) / gpu_data->lane_pool.max_density[lane_index] * ( gpu_data->lane_pool.max_density[lane_index] - 0 );

	if (speed_ < data_setting_gpu->kOnGPUMinSpeed)
		speed_ = data_setting_gpu->kOnGPUMinSpeed;

//update speed history
	gpu_data->lane_pool.speed_history[time_index][lane_index] = speed_;

	gpu_data->lane_pool.density[lane_index] = density_;
	gpu_data->lane_pool.speed[lane_index] = speed_;
//estimated empty_space

	float prediction_queue_length_ = 0.0f;

	if (time_step < data_setting_gpu->kOnGPUStartTimeStep + 4 * data_setting_gpu->kOnGPUUnitTimeStep) {
//		gpu_data->lane_pool.predicted_empty_space[lane_index] = gpu_data->lane_pool.his_queue_length[0][lane_index];
//		gpu_data->lane_pool.predicted_queue_length[lane_index] = 0;

		gpu_data->lane_pool.predicted_empty_space[lane_index] = MinOnDevice(
				gpu_data->lane_pool.last_time_empty_space[lane_index] + (gpu_data->lane_pool.speed[lane_index] * data_setting_gpu->kOnGPUUnitTimeStep), 1.0f * gpu_data->lane_pool.lane_length[lane_index]);
	} else {
		prediction_queue_length_ = gpu_data->lane_pool.his_queue_length[0][lane_index];
		prediction_queue_length_ += (gpu_data->lane_pool.his_queue_length[0][lane_index] - gpu_data->lane_pool.his_queue_length[1][lane_index])
				* gpu_data->lane_pool.his_queue_length_weighting[0][lane_index];

//		prediction_empty_space_ += (gpu_data->lane_pool.his_queue_length[1][lane_index] - gpu_data->lane_pool.his_queue_length[2][lane_index])
//				* gpu_data->lane_pool.his_queue_length_weighting[1][lane_index];
//
//		prediction_empty_space_ += (gpu_data->lane_pool.his_queue_length[2][lane_index] - gpu_data->lane_pool.his_queue_length[3][lane_index])
//				* gpu_data->lane_pool.his_queue_length_weighting[2][lane_index];

		gpu_data->lane_pool.predicted_empty_space[lane_index] = MinOnDevice(
				gpu_data->lane_pool.last_time_empty_space[lane_index] + (gpu_data->lane_pool.speed[lane_index] * data_setting_gpu->kOnGPUUnitTimeStep),
				(gpu_data->lane_pool.lane_length[lane_index] - prediction_queue_length_));
	}

//	gpu_data->lane_pool.debug_data[lane_index] = gpu_data->lane_pool.predicted_empty_space[lane_index];
//update Tp

	gpu_data->lane_pool.accumulated_offset[lane_index] += gpu_data->lane_pool.speed[lane_index] * data_setting_gpu->kOnGPUUnitTimeStep; //meter

	while (gpu_data->lane_pool.accumulated_offset[lane_index] >= gpu_data->lane_pool.lane_length[lane_index]) {
		gpu_data->lane_pool.accumulated_offset[lane_index] -= gpu_data->lane_pool.speed_history[gpu_data->lane_pool.Tp[lane_index]][lane_index] * data_setting_gpu->kOnGPUUnitTimeStep;
		gpu_data->lane_pool.Tp[lane_index] += data_setting_gpu->kOnGPUUnitTimeStep;
	}

	//update queue length
	int queue_start = gpu_data->lane_pool.queue_length[lane_index] / data_setting_gpu->kOnGPUVehicleLength;
	for (int queue_index = queue_start; queue_index < gpu_data->lane_pool.vehicle_counts[lane_index]; queue_index++) {
		int vehicle_index = gpu_data->lane_vehicle_pool.vehicle_space[gpu_data->lane_pool.vehicle_start_index[lane_index] + queue_index];
//		if (gpu_data->lane_pool.vehicle_space[queue_index][lane_index]->entry_time <= gpu_data->lane_pool.Tp[lane_index]) {
		if (vpool_gpu[vehicle_index].entry_time <= gpu_data->lane_pool.Tp[lane_index]) {
			gpu_data->lane_pool.queue_length[lane_index] += data_setting_gpu->kOnGPUVehicleLength;
		} else {
			break;
		}
	}
}

__device__ int GetNextVehicleAtNode(GPUMemory* gpu_data, int node_index, int* lane_index, GPUSharedParameter* data_setting_gpu, GPUVehicle *vpool_gpu) {

	int maximum_waiting_time = -1;
//	int the_lane_index = -1;
	int the_one_veh = -1;

	int upstream_start_lane = gpu_data->node_pool.upstream_lane_start_index[node_index];
	int upstream_end_lane = gpu_data->node_pool.upstream_lane_end_index[node_index];

	//no upstream links, so, return -1, no vehicle
	if(upstream_start_lane < 0 || upstream_end_lane < 0) return -1;

	for (int one_lane_index = upstream_start_lane; one_lane_index <= upstream_end_lane; one_lane_index++) {
		/*
		 * Condition 1: The Lane is not NULL
		 * ----      2: Has Output Capacity
		 * ---       3: Is not blocked
		 * ---       4: Has vehicles
		 * ---       5: The vehicle can pass
		 */

		if (gpu_data->lane_pool.output_capacity[one_lane_index] > 0 && gpu_data->lane_pool.blocked[one_lane_index] == false && gpu_data->lane_pool.vehicle_counts[one_lane_index] > 0) {
//			int start_vehicle_index = gpu_data->lane_pool.vehicle_start_index;
//			int end_vehicle_index = gpu_data->lane_pool.vehicle_end_index;

			int first_vehicle_ID = gpu_data->lane_vehicle_pool.vehicle_space[gpu_data->lane_pool.vehicle_start_index[one_lane_index]];

//			int time_diff = gpu_data->lane_pool.Tp[one_lane_index] - gpu_data->lane_pool.vehicle_space[0][one_lane_index]->entry_time;
			int time_diff = gpu_data->lane_pool.Tp[one_lane_index] - vpool_gpu[first_vehicle_ID].entry_time;
			if (time_diff >= 0) {

				//if already the final move, then no need for checking next road
				if ((vpool_gpu[first_vehicle_ID].next_path_index) >= (vpool_gpu[first_vehicle_ID].whole_path_length)) {
					if (time_diff > maximum_waiting_time) {
						maximum_waiting_time = time_diff;
						*lane_index = one_lane_index;
						the_one_veh = first_vehicle_ID;
//						return gpu_data->lane_pool.vehicle_space[0][one_lane_index];
					}
				} else {
					int next_lane_index = vpool_gpu[first_vehicle_ID].path_code[vpool_gpu[first_vehicle_ID].next_path_index];

					/**
					 * Condition 6: The Next Lane has input capacity
					 * ---       7: The next lane has empty space
					 */
					if (gpu_data->lane_pool.input_capacity[next_lane_index] > 0 && gpu_data->lane_pool.predicted_empty_space[next_lane_index] > data_setting_gpu->kOnGPUVehicleLength) {
						if (time_diff > maximum_waiting_time) {
							maximum_waiting_time = time_diff;
							*lane_index = one_lane_index;
							the_one_veh = first_vehicle_ID;
//								return gpu_data->lane_pool.vehicle_space[0][one_lane_index];
						}
					} else {
						gpu_data->lane_pool.blocked[one_lane_index] = true;
					}
				}
			}
		}
	}

	return the_one_veh;
}

__global__ void SupplySimulationVehiclePassing(GPUMemory* gpu_data, int time_step, int node_length, GPUSharedParameter* data_setting_gpu, GPUVehicle *vpool_gpu) {
	int node_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (node_index >= node_length)
		return;

	for (int i = 0; i < gpu_data->node_pool.max_acc_flow[node_index]; i++) {
		int lane_index = -1;

		//Find A vehicle
		int vehicle_passing_index = GetNextVehicleAtNode(gpu_data, node_index, &lane_index, data_setting_gpu, vpool_gpu);

		if (vehicle_passing_index < 0 || lane_index < 0) {
			//			printf("one_v == NULL\n");
			break;
		}

		if (vpool_gpu[vehicle_passing_index].entry_time <= gpu_data->lane_pool.Tp[lane_index]) {
			gpu_data->lane_pool.queue_length[lane_index] -= data_setting_gpu->kOnGPUVehicleLength;
		}

		//Insert to next Lane
//		if (gpu_data->lane_pool.vehicle_space[0][lane_index]->next_path_index >= gpu_data->lane_pool.vehicle_space[0][lane_index]->whole_path_length) {
		int vehicle_index = gpu_data->lane_vehicle_pool.vehicle_space[gpu_data->lane_pool.vehicle_start_index[lane_index]];
		if (vpool_gpu[vehicle_index].next_path_index >= vpool_gpu[vehicle_index].whole_path_length) {
			//the vehicle has finished the trip

			//			printf("vehicle %d finish trip at node %d,\n", one_v->vehicle_ID, node_index);
		} else {
			int next_lane_index = vpool_gpu[vehicle_index].path_code[vpool_gpu[vehicle_index].next_path_index];
			vpool_gpu[vehicle_index].next_path_index++;

			//it is very critical to update the entry time when passing
			vpool_gpu[vehicle_index].entry_time = time_step;

			//add the vehicle
//			gpu_data->lane_pool.vehicle_passed_space[gpu_data->lane_pool.vehicle_passed_to_the_lane_counts[next_lane_index]][next_lane_index] = vehicle_passing_index;
			int buffer_vehicle_index = gpu_data->lane_pool.buffered_vehicle_start_index[next_lane_index] + gpu_data->lane_pool.vehicle_passed_to_the_lane_counts[next_lane_index];
			gpu_data->lane_buffered_vehicle_pool.buffered_vehicle_space[buffer_vehicle_index] = vehicle_passing_index;
			gpu_data->lane_pool.vehicle_passed_to_the_lane_counts[next_lane_index]++;

			gpu_data->lane_pool.input_capacity[next_lane_index]--;
			gpu_data->lane_pool.predicted_empty_space[next_lane_index] -= data_setting_gpu->kOnGPUVehicleLength;

			//printf("time_step=%d,one_v->vehicle_ID=%d,lane_index=%d, next_lane_index=%d, next_lane_index=%d\n", time_step, one_v->vehicle_ID, lane_index, next_lane_index, next_lane_index);
		}

		//Remove from current Lane
		int start_vehicle_pool_index = gpu_data->lane_pool.vehicle_start_index[lane_index];
		for (int j = 1; j < gpu_data->lane_pool.vehicle_counts[lane_index]; j++) {
			gpu_data->lane_vehicle_pool.vehicle_space[start_vehicle_pool_index + j-1] = gpu_data->lane_vehicle_pool.vehicle_space[start_vehicle_pool_index + j];
		}

		gpu_data->lane_pool.vehicle_counts[lane_index]--;
		gpu_data->lane_pool.output_capacity[lane_index]--;
		gpu_data->lane_pool.flow[lane_index]++;
	}
}


__global__ void SupplySimulationAfterVehiclePassing(GPUMemory* gpu_data, int time_step, int segment_length, GPUSharedParameter* data_setting_gpu) {
	int lane_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (lane_index >= segment_length)
		return;

//update queue length
//	bool continue_loop = true;
//	float queue_length = 0;
//	float acc_length_moving = gpu_data->lane_pool.accumulated_offset[lane_index];
//	int to_time_step = gpu_data->lane_pool.Tp[lane_index];
//
//	for (int i = 0; continue_loop && i < gpu_data->lane_pool.vehicle_counts[lane_index]; i++) {
//		if (gpu_data->lane_pool.vehicle_space[i][lane_index]->entry_time <= gpu_data->lane_pool.Tp[lane_index]) {
//			queue_length += data_setting_gpu->kOnGPUVehicleLength;
//		}
//		else {
//			int entry_time = gpu_data->lane_pool.vehicle_space[i][lane_index]->entry_time;
//			for (int j = entry_time; i < to_time_step; i++) {
//				acc_length_moving -= gpu_data->lane_pool.speed_history[j][lane_index] * data_setting_gpu->kOnGPUUnitTimeStep;
//			}
//
//			if (acc_length_moving + queue_length >= gpu_data->lane_pool.lane_length[lane_index]) {
//				to_time_step = entry_time;
//				queue_length += data_setting_gpu->kOnGPUVehicleLength;
//			}
//			else {
//				continue_loop = false;
//			}
//		}
//	}
//
////update queue length
//	gpu_data->lane_pool.queue_length[lane_index] = queue_length;

//update the queue history
	for (int i = 3; i > 0; i--) {
		gpu_data->lane_pool.his_queue_length[i][lane_index] = gpu_data->lane_pool.his_queue_length[i - 1][lane_index];
	}
	gpu_data->lane_pool.his_queue_length[0][lane_index] = gpu_data->lane_pool.queue_length[lane_index];

//update the empty space
//			if (gpu_data->lane_pool.new_vehicle_join_counts[lane_index] > 0) {
//				gpu_data->lane_pool.empty_space[lane_index] = std::min(gpu_data->lane_pool.speed[lane_index] * UNIT_TIME_STEPS, gpu_data->lane_pool.empty_space[lane_index])
//				if (gpu_data->lane_pool.empty_space[lane_index] < 0) gpu_data->lane_pool.empty_space[lane_index] = 0;
//			}
//			else {
	gpu_data->lane_pool.empty_space[lane_index] = gpu_data->lane_pool.empty_space[lane_index] + gpu_data->lane_pool.speed[lane_index] * data_setting_gpu->kOnGPUUnitTimeStep;
	gpu_data->lane_pool.empty_space[lane_index] = MinOnDevice(gpu_data->lane_pool.lane_length[lane_index] - gpu_data->lane_pool.queue_length[lane_index], gpu_data->lane_pool.empty_space[lane_index]);

}

__global__ void supply_simulated_results_to_buffer(GPUMemory* gpu_data, int time_step, int segment_length, SimulationResults* buffer, GPUSharedParameter* data_setting_gpu) {
	int buffer_index = time_step % data_setting_gpu->kOnGPUGPUToCPUSimulationResultsCopyBufferSize;

	int lane_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (lane_index >= segment_length)
		return;

	buffer[buffer_index].flow[lane_index] = gpu_data->lane_pool.flow[lane_index];
	buffer[buffer_index].density[lane_index] = gpu_data->lane_pool.density[lane_index];
	buffer[buffer_index].speed[lane_index] = gpu_data->lane_pool.speed[lane_index];
	buffer[buffer_index].queue_length[lane_index] = gpu_data->lane_pool.queue_length[lane_index];
	buffer[buffer_index].counts[lane_index] = gpu_data->lane_pool.vehicle_counts[lane_index];

}

__device__ float MinOnDevice(float one_value, float the_other) {
	if (one_value < the_other)
		return one_value;
	else
		return the_other;
}

__device__ float MaxOnDevice(float one_value, float the_other) {
	if (one_value > the_other)
		return one_value;
	else
		return the_other;
}
#endif /* ONGPU_KERNAL_H_ */
