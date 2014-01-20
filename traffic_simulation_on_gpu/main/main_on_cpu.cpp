///**
// *This project targets to check GPU is an option for DynaMIT.
// *This project also targets for a paper "Mesoscopic Traffic Simulation on GPU"
// */
//
//#include "../on_cpu/network/Network.h"
//#include "../on_cpu/demand/OD_Pair.h"
//#include "../on_cpu/demand/OD_Path.h"
//#include "../on_cpu/demand/Vehicle.h"
//#include "../on_cpu/util/TimeTools.h"
//
//#include "../on_cpu/util/shared_cpu_include.h"
//#include "../on_gpu/supply/kernel_functions.h"
//#include "../on_gpu/supply/OnGPUMemory.h"
//#include "../on_cpu/util/SimulationResults.h"
//#include "../on_gpu/supply/OnGPUVehicle.h"
//#include "../on_gpu/supply/OnGPUNewLaneVehicles.h"
//
//#define ENABLE_OUTPUT
//
//using namespace std;
//
///*
// * Demand
// */
//Network* the_network;
//vector<OD_Pair*> all_od_pairs;
//vector<OD_Pair_PATH*> all_od_paths;
//vector<Vehicle*> all_vehicles;
//
///*
// * Path Input Config
// */
//std::string network_file_path = "data/network_100.dat";
//std::string demand_file_path = "data/demand_100_10000.dat";
//std::string od_pair_file_path = "data/od_pair_100.dat";
//std::string od_pair_paths_file_path = "data/od_pair_paths_100.dat";
//
///*
// * All data in GPU
// */
//GPUMemory* gpu_data;
//
////A large memory space is pre-defined in order to copy to GPU
//GPUVehicle *vpool_cpu;
//
///**
// * Simulation Results
// */
//std::string simulation_output_file_path = "output/simulated_outputs.txt";
//std::map<int, SimulationResults*> simulation_results_pool;
//ofstream simulation_results_output_file;
//
///*
// * GPU Streams
// * stream1: GPU Supply Simulation
// */
//cudaStream_t stream_gpu_supply;
//cudaEvent_t GPU_supply_one_time_simulation_done_event;
//
///*
// * Time Management
// */
//long simulation_start_time;
//long simulation_end_time;
//long simulation_time_step;
//
///*
// * simulation_time is already finished time;
// * simulation_time + 1 might be the current simulating time on GPU
// */
//long to_simulate_time;
//
///*
// * simulation_results_outputed_time is already outputted time;
// * simulation_results_outputed_time + 1 might be the outputing time on CPU
// */
//long to_output_simulation_result_time;
//
///*
// * Define Major Functions
// */
//bool load_in_network();
//bool load_in_demand();
//bool initilizeCPU();
//bool initilizeGPU();
//bool initGPUData(GPUMemory* data_local);
//
//bool start_simulation();
//bool destory_resources();
//
///*
// * Define Helper Functions
// */
//bool copy_simulated_results_to_CPU(int time_step);
//bool output_simulated_results(int time_step);
//
//inline int timestep_to_arrayindex(int time_step) {
//	return (time_step - START_TIME_STEPS) / UNIT_TIME_STEPS;
//}
//
///*
// * Supply Function Define
// */
//void supply_simulation_pre_vehicle_passing(int time_step);
//void supply_simulation_vehicle_passing(int time_step);
//void supply_simulation_after_vehicle_passing(int time_step);
//
//GPUVehicle* get_next_vehicle_at_node(int node_id, int* lane_id);
//
///*
// * MAIN
// */
//int main() {
//	if (load_in_network() == false) {
//		cout << "Loading network fails" << endl;
//		return 0;
//	}
//
//	if (load_in_demand() == false) {
//		cout << "Loading demand fails" << endl;
//		return 0;
//	}
//
//	if (initilizeCPU() == false) {
//		cout << "InitilizeCPU fails" << endl;
//		return 0;
//	}
//
//	if (initilizeGPU() == false) {
//		cout << "InitilizeGPU fails" << endl;
//		return 0;
//	}
//
//	TimeTools profile;
//	profile.start_profiling();
//
//	std::cout << "Simulation Starts" << std::endl;
//
//	//Start Simulation
//	if (start_simulation() == false) {
//		cout << "Simulation fails" << endl;
//		destory_resources();
//		return 0;
//	}
//
//	profile.end_profiling();
//	profile.output();
//
//	cout << "Simulation Succeed!" << endl;
//
//	destory_resources();
//
//	cout << "Simulation Succeed!" << endl;
//
//	return 0;
//}
//
///**
// *
// */
//bool load_in_network() {
//	the_network = new Network();
//
//	the_network->all_links.clear();
//	the_network->all_nodes.clear();
//	the_network->node_mapping.clear();
//
//	return Network::load_network(the_network, network_file_path);
//}
//
//bool load_in_demand() {
//
//	if (OD_Pair::load_in_all_ODs(all_od_pairs, od_pair_file_path) == false) {
//		return false;
//	}
//
//	if (OD_Pair_PATH::load_in_all_OD_Paths(all_od_paths, od_pair_paths_file_path) == false) {
//		return false;
//	}
//
//	if (Vehicle::load_in_all_vehicles(all_vehicles, demand_file_path) == false) {
//		return false;
//	}
//
//	return true;
//}
//
//bool initilizeCPU() {
//	simulation_start_time = START_TIME_STEPS;
//	simulation_end_time = END_TIME_STEPS; // 2 hours
//	simulation_time_step = UNIT_TIME_STEPS;
//
//	assert(simulation_time_step == 1);
//
//	to_simulate_time = simulation_start_time;
//	to_output_simulation_result_time = simulation_start_time;
//
//	simulation_results_pool.clear();
//	simulation_results_output_file.open(simulation_output_file_path.c_str());
//	simulation_results_output_file << "##TIME STEP" << ":Lane ID:" << ":(" << "COUNTS" << ":" << "flow" << ":" << "density" << ":" << "speed" << ":" << "queue_length" << ")" << endl;
//
//	return true;
//}
//
//bool initilizeGPU() {
//	gpu_data = new GPUMemory();
//	initGPUData(gpu_data);
//	return true;
//}
//
///*
// * Build a GPU data
// */
//bool initGPUData(GPUMemory* data_local) {
//
//	/**
//	 * First Part: Lane
//	 */
//
//	for (int i = 0; i < the_network->all_links.size(); i++) {
//		Link* one_link = the_network->all_links[i];
//
//		data_local->lane_pool.lane_ID[i] = one_link->link_id;
//		//make sure assert is working
////		assert(1 == 0);
//
//		assert(one_link->link_id == i);
//
//		data_local->lane_pool.from_node_id[i] = one_link->from_node->node_id;
//		data_local->lane_pool.to_node_id[i] = one_link->to_node->node_id;
//
//		data_local->lane_pool.Tp[i] = simulation_start_time - simulation_time_step;
//		data_local->lane_pool.Tq[i] = simulation_start_time - simulation_time_step;
//		data_local->lane_pool.accumulated_offset[i] = 0;
//
//		data_local->lane_pool.flow[i] = 0;
//		data_local->lane_pool.density[i] = 0;
//		data_local->lane_pool.speed[i] = 0;
//		data_local->lane_pool.queue_length[i] = 0;
//
//		/*
//		 * for density calculation
//		 */
//		data_local->lane_pool.lane_length[i] = ROAD_LENGTH; // meter
//		data_local->lane_pool.max_vehicles[i] = ROAD_LENGTH / VEHICLE_LENGTH; //number of vehicles
//		data_local->lane_pool.output_capacity[i] = LANE_OUTPUT_CAPACITY_TIME_STEP; //
//		data_local->lane_pool.input_capacity[i] = LANE_INPUT_CAPACITY_TIME_STEP; //
//		data_local->lane_pool.empty_space[i] = ROAD_LENGTH;
//
//		/*
//		 * for speed calculation
//		 */
//		data_local->lane_pool.alpha[i] = Alpha;
//		data_local->lane_pool.beta[i] = Beta;
//		data_local->lane_pool.max_density[i] = Max_Density;
//		data_local->lane_pool.min_density[i] = Min_Density;
//		data_local->lane_pool.MAX_SPEED[i] = MAX_SPEED;
//		data_local->lane_pool.MIN_SPEED[i] = MIN_SPEED;
//
//		data_local->lane_pool.vehicle_counts[i] = 0;
//		data_local->lane_pool.vehicle_passed_to_the_lane_counts[i] = 0;
//
//		for (int c = 0; c < MAX_VEHICLE_PER_LANE; c++) {
//			data_local->lane_pool.vehicle_space[c][i] = NULL;
//		}
//
//		for (int c = 0; c < LANE_INPUT_CAPACITY_TIME_STEP; c++) {
//			data_local->lane_pool.vehicle_passed_space[c][i] = NULL;
//		}
//
//		for (int j = 0; j < TOTAL_TIME_STEPS; j++) {
//			data_local->lane_pool.speed_history[j][i] = -1;
//		}
//
//		//it is assumed that QUEUE_LENGTH_HISTORY = 4;
//		assert(QUEUE_LENGTH_HISTORY == 4);
//		float weight[QUEUE_LENGTH_HISTORY];
//		weight[0] = 0.5;
//		weight[1] = 0.3;
//		weight[2] = 0.2;
//		weight[3] = 0;
//
//		//		{ 0.2, 0.3, 0.5, 0 };
//
//		for (int j = 0; j < QUEUE_LENGTH_HISTORY; j++) {
//			data_local->lane_pool.his_queue_length[j][i] = -1;
//			data_local->lane_pool.his_queue_length_weighting[j][i] = weight[j];
//		}
//
//		data_local->lane_pool.predicted_empty_space[i] = 0;
//		data_local->lane_pool.predicted_queue_length[i] = 0;
//	}
//
//	/**
//	 * Second Part: Node
//	 */
////	NodePool* the_node_pool = data_local->node_pool;
//	for (int i = 0; i < the_network->all_nodes.size(); i++) {
//		Node* one_node = the_network->all_nodes[i];
//
//		data_local->node_pool.node_ID[i] = one_node->node_id;
//		data_local->node_pool.MAXIMUM_ACCUMULATED_FLOW[i] = 0;
//		data_local->node_pool.ACCUMULATYED_UPSTREAM_CAPACITY[i] = 0;
//		data_local->node_pool.ACCUMULATYED_DOWNSTREAM_CAPACITY[i] = 0;
//
//		assert(one_node->node_id == i);
//
//		for (int j = 0; j < MAX_LANE_UPSTREAM; j++) {
//			data_local->node_pool.upstream[j][i] = -1;
//		}
//
//		for (int j = 0; j < one_node->upstream_links.size(); j++) {
//			data_local->node_pool.upstream[j][i] = one_node->upstream_links[j]->link_id;
//			data_local->node_pool.ACCUMULATYED_UPSTREAM_CAPACITY[i] += LANE_OUTPUT_CAPACITY_TIME_STEP;
//		}
//
//		for (int j = 0; j < MAX_LANE_DOWNSTREAM; j++) {
//			data_local->node_pool.downstream[j][i] = -1;
//		}
//
//		for (int j = 0; j < one_node->downstream_links.size(); j++) {
//			data_local->node_pool.downstream[j][i] = one_node->downstream_links[j]->link_id;
//			data_local->node_pool.ACCUMULATYED_DOWNSTREAM_CAPACITY[i] += LANE_OUTPUT_CAPACITY_TIME_STEP;
//		}
//
//		data_local->node_pool.MAXIMUM_ACCUMULATED_FLOW[i] =
//				(data_local->node_pool.ACCUMULATYED_UPSTREAM_CAPACITY[i] < data_local->node_pool.ACCUMULATYED_DOWNSTREAM_CAPACITY[i]) ?
//						data_local->node_pool.ACCUMULATYED_UPSTREAM_CAPACITY[i] : data_local->node_pool.ACCUMULATYED_DOWNSTREAM_CAPACITY[i];
//
////		std::cout << "MAXIMUM_ACCUMULATED_FLOW:" << i << ", " << data_local->node_pool.MAXIMUM_ACCUMULATED_FLOW[i] << std::endl;
//	}
//
//	/**
//	 * Third Part:
//	 */
//
////Init VehiclePool
//	for (int i = START_TIME_STEPS; i < TOTAL_TIME_STEPS; i += UNIT_TIME_STEPS) {
//		for (int j = 0; j < LANE_SIZE; j++) {
//			data_local->new_vehicles_every_time_step[i].new_vehicle_size[j] = 0;
//			data_local->new_vehicles_every_time_step[i].lane_ID[j] = -1;
//		}
//	}
//
//	std::cout << "all_vehicles.size():" << all_vehicles.size() << std::endl;
//
//	std::cout << "TOTAL_TIME_STEPS * LANE_SIZE * VEHICLE_MAX_LOADING_ONE_TIME * sizeof(GPUVehicle):" << TOTAL_TIME_STEPS * LANE_SIZE * VEHICLE_MAX_LOADING_ONE_TIME * sizeof(GPUVehicle) << std::endl;
//
//	std::cout << "TOTAL_TIME_STEPS * LANE_SIZE * VEHICLE_MAX_LOADING_ONE_TIME * sizeof(GPUVehicle):" << TOTAL_TIME_STEPS * LANE_SIZE * VEHICLE_MAX_LOADING_ONE_TIME * sizeof(GPUVehicle) << std::endl;
//
////init host vehicle pool data /*xiaosong*/
//	int memory_space_for_vehicles = all_vehicles.size() * sizeof(GPUVehicle);
//	vpool_cpu = (GPUVehicle*) malloc(memory_space_for_vehicles);
//	if (vpool_cpu == NULL) exit(1);
//
//	std::cout << "TOTAL_TIME_STEPS * LANE_SIZE * VEHICLE_MAX_LOADING_ONE_TIME * sizeof(GPUVehicle):" << TOTAL_TIME_STEPS * LANE_SIZE * VEHICLE_MAX_LOADING_ONE_TIME * sizeof(GPUVehicle) << std::endl;
//
//	int nVehiclePerTick = VEHICLE_MAX_LOADING_ONE_TIME * LANE_SIZE;
//
////	std::cout << "total array size:" << (TOTAL_TIME_STEPS * LANE_SIZE * VEHICLE_MAX_LOADING_ONE_TIME) << std::endl;
////	std::cout << "total size:" << (TOTAL_TIME_STEPS * LANE_SIZE * VEHICLE_MAX_LOADING_ONE_TIME * sizeof(GPUVehicle)) << std::endl;
//
//	std::cout << "init all_vehicles" << std::endl;
//	int total_inserted_vehicles = 0;
//
////Insert Vehicles
//	for (int i = 0; i < all_vehicles.size(); i++) {
//		Vehicle* one_vehicle = all_vehicles[i];
//
//		int time_index = one_vehicle->entry_time;
//		int time_index_covert = timestep_to_arrayindex(time_index);
//		assert(time_index == time_index_covert);
//		//try to load vehicles beyond the simulation border
//		if (time_index_covert >= TOTAL_TIME_STEPS) continue;
//
//		int lane_ID = all_od_paths[one_vehicle->path_id]->link_ids[0];
//
//		if (data_local->new_vehicles_every_time_step[time_index_covert].new_vehicle_size[lane_ID] < VEHICLE_MAX_LOADING_ONE_TIME) {
//			int last_vehicle_index = data_local->new_vehicles_every_time_step[time_index_covert].new_vehicle_size[lane_ID];
//			int idx_vpool = time_index_covert * nVehiclePerTick + lane_ID * VEHICLE_MAX_LOADING_ONE_TIME + last_vehicle_index;
//
//			//for gpu to rebuild
////			vpool_cpu_index[idx_vpool] = i;
////			std::cout << "idx_vpool:" << idx_vpool << " is map to i:" << i << std::endl;
//
//			vpool_cpu[i].vehicle_ID = one_vehicle->vehicle_id;
//			vpool_cpu[i].entry_time = time_index;
//			vpool_cpu[i].current_lane_ID = lane_ID;
//			int max_copy_length = MAX_ROUTE_LENGTH > all_od_paths[one_vehicle->path_id]->link_ids.size() ? all_od_paths[one_vehicle->path_id]->link_ids.size() : MAX_ROUTE_LENGTH;
//
//			for (int p = 1; p < max_copy_length; p++) {
//				vpool_cpu[i].path_code[p - 1] = all_od_paths[one_vehicle->path_id]->route_code[p] ? 1 : 0;
//			}
//
//			//ready for the next lane, so next_path_index is set to 1, if the next_path_index == whole_path_length, it means cannot find path any more, can exit;
//			vpool_cpu[i].next_path_index = 1;
//			vpool_cpu[i].whole_path_length = all_od_paths[one_vehicle->path_id]->link_ids.size();
//
//			data_local->new_vehicles_every_time_step[time_index_covert].new_vehicles[lane_ID][last_vehicle_index] = &(vpool_cpu[i]);
//			data_local->new_vehicles_every_time_step[time_index_covert].new_vehicle_size[lane_ID]++;
//
//			total_inserted_vehicles++;
//		}
//		else {
//			std::cout << "Loading Vehicles Exceeds The Loading Capacity: Time:" << time_index_covert << ", Lane_ID:" << lane_ID << ",i:" << i << ",ID:" << one_vehicle->vehicle_id << std::endl;
//		}
//	}
//
//	std::cout << "init all_vehicles done" << total_inserted_vehicles << std::endl;
//
//	return true;
//}
//
//bool destory_resources() {
//	simulation_results_output_file.flush();
//	simulation_results_output_file.close();
//
////	cudaEventDestroy(GPU_supply_one_time_simulation_done_event);
////	cudaStreamDestroy(stream_gpu_supply);
//
//	if (vpool_cpu != NULL) delete vpool_cpu;
////	if (vpool_cpu_index != NULL) delete vpool_cpu_index;
//
////	cudaDeviceReset();
//	return true;
//}
//
//bool start_simulation() {
//
//	while (((to_simulate_time >= simulation_end_time) && (to_output_simulation_result_time >= simulation_end_time)) == false) {
//
//		std::cout << "Current Time: " << to_simulate_time << std::endl;
//
//		supply_simulation_pre_vehicle_passing(to_simulate_time);
//
//		supply_simulation_vehicle_passing(to_simulate_time);
//
//		supply_simulation_after_vehicle_passing(to_simulate_time);
//
//		output_simulated_results(to_output_simulation_result_time);
//
//		to_simulate_time += simulation_time_step;
//		to_output_simulation_result_time += simulation_time_step;
//	}
//
//	return true;
//}
//
///**
// * Minor Functions
// */
//
//bool output_simulated_results(int time_step) {
//
//	for (int i = 0; i < LANE_SIZE; i++) {
//		simulation_results_output_file << time_step << ":lane:" << i << ":(" << gpu_data->lane_pool.vehicle_counts[i] << ":" << gpu_data->lane_pool.flow[i] << ":" << gpu_data->lane_pool.density[i]
//				<< ":" << gpu_data->lane_pool.speed[i] << ":" << gpu_data->lane_pool.queue_length[i] << ")" << endl;
//	}
//
//	return true;
//}
//
///**
// * Kernel Functions, not sure how to move to other folder
// */
//
///*
// * Supply Function Implementation
// */
//
//void supply_simulation_pre_vehicle_passing(int time_step) {
//	int time_index = time_step;
//
//	for (unsigned int lane_id = 0; lane_id < the_network->all_links.size(); lane_id++) {
//
//		gpu_data->lane_pool.new_vehicle_join_counts[lane_id] = 0;
//
////init capacity
//		gpu_data->lane_pool.input_capacity[lane_id] = LANE_INPUT_CAPACITY_TIME_STEP;
//		gpu_data->lane_pool.output_capacity[lane_id] = LANE_OUTPUT_CAPACITY_TIME_STEP;
//
////init for next GPU kernel function
//		gpu_data->lane_pool.blocked[lane_id] = false;
//
////load passed vehicles to the back of the lane
//		for (int i = 0; i < gpu_data->lane_pool.vehicle_passed_to_the_lane_counts[lane_id]; i++) {
//			if (gpu_data->lane_pool.vehicle_counts[lane_id] < gpu_data->lane_pool.max_vehicles[lane_id]) {
//				gpu_data->lane_pool.vehicle_space[gpu_data->lane_pool.vehicle_counts[lane_id]][lane_id] = gpu_data->lane_pool.vehicle_passed_space[i][lane_id];
//				gpu_data->lane_pool.vehicle_counts[lane_id]++;
//
//				gpu_data->lane_pool.new_vehicle_join_counts[lane_id]++;
//			}
//		}
//		gpu_data->lane_pool.vehicle_passed_to_the_lane_counts[lane_id] = 0;
//
////
////load newly generated vehicles to the back of the lane
//		for (int i = 0; i < gpu_data->new_vehicles_every_time_step[time_index].new_vehicle_size[lane_id]; i++) {
//			if (gpu_data->lane_pool.vehicle_counts[lane_id] < gpu_data->lane_pool.max_vehicles[lane_id]) {
//				gpu_data->lane_pool.vehicle_space[gpu_data->lane_pool.vehicle_counts[lane_id]][lane_id] = (gpu_data->new_vehicles_every_time_step[time_index].new_vehicles[lane_id][i]);
//				gpu_data->lane_pool.vehicle_counts[lane_id]++;
//
//				gpu_data->lane_pool.new_vehicle_join_counts[lane_id]++;
//			}
//		}
//
////update speed and density
//		gpu_data->lane_pool.density[lane_id] = 1.0 * VEHICLE_LENGTH * gpu_data->lane_pool.vehicle_counts[lane_id] / gpu_data->lane_pool.lane_length[lane_id];
//
////		std::cout << "gpu_data->lane_pool.speed[lane_id]:" << gpu_data->lane_pool.speed[lane_id] << std::endl;
//
////Speed-Density Relationship
//		gpu_data->lane_pool.speed[lane_id] = gpu_data->lane_pool.MAX_SPEED[lane_id]
//				* (pow((1 - pow((gpu_data->lane_pool.density[lane_id] / gpu_data->lane_pool.max_density[lane_id]), gpu_data->lane_pool.beta[lane_id])), gpu_data->lane_pool.alpha[lane_id]));
//
//		if (gpu_data->lane_pool.speed[lane_id] < gpu_data->lane_pool.MIN_SPEED[lane_id]) gpu_data->lane_pool.speed[lane_id] = gpu_data->lane_pool.MIN_SPEED[lane_id];
//
////		std::cout << "gpu_data->lane_pool.speed[lane_id]:" << gpu_data->lane_pool.speed[lane_id] << std::endl;
//
////update speed history
//		gpu_data->lane_pool.speed_history[time_index][lane_id] = gpu_data->lane_pool.speed[lane_id];
//
////estimated empty_space
//		if (time_step < START_TIME_STEPS + 4 * UNIT_TIME_STEPS) {
////		gpu_data->lane_pool.predicted_empty_space[lane_id] = gpu_data->lane_pool.his_queue_length[0][lane_id];
//			gpu_data->lane_pool.predicted_queue_length[lane_id] = 0;
//			gpu_data->lane_pool.predicted_empty_space[lane_id] = ROAD_LENGTH;
//		}
//		else {
//			gpu_data->lane_pool.predicted_queue_length[lane_id] = gpu_data->lane_pool.his_queue_length[0][lane_id];
//			gpu_data->lane_pool.predicted_queue_length[lane_id] += (gpu_data->lane_pool.his_queue_length[0][lane_id] - gpu_data->lane_pool.his_queue_length[1][lane_id])
//					* gpu_data->lane_pool.his_queue_length_weighting[0][lane_id];
//
//			gpu_data->lane_pool.predicted_queue_length[lane_id] += (gpu_data->lane_pool.his_queue_length[1][lane_id] - gpu_data->lane_pool.his_queue_length[2][lane_id])
//					* gpu_data->lane_pool.his_queue_length_weighting[1][lane_id];
//
//			gpu_data->lane_pool.predicted_queue_length[lane_id] += (gpu_data->lane_pool.his_queue_length[2][lane_id] - gpu_data->lane_pool.his_queue_length[3][lane_id])
//					* gpu_data->lane_pool.his_queue_length_weighting[2][lane_id];
//
//			//need improve
//			//XUYAN, need modify
//			gpu_data->lane_pool.predicted_empty_space[lane_id] = (ROAD_LENGTH - gpu_data->lane_pool.predicted_queue_length[lane_id]);
//		}
//
////		std::cout << "gpu_data->lane_pool.accumulated_offset[lane_id]:" << gpu_data->lane_pool.accumulated_offset[lane_id] << std::endl;
//
////update Tp
//		gpu_data->lane_pool.accumulated_offset[lane_id] += gpu_data->lane_pool.speed[lane_id] * UNIT_TIME_STEPS; //meter
//
////		std::cout << "gpu_data->lane_pool.accumulated_offset[lane_id]:" << gpu_data->lane_pool.accumulated_offset[lane_id] << std::endl;
////		std::cout << "gpu_data->lane_pool.speed[lane_id]:" << gpu_data->lane_pool.speed[lane_id] << std::endl;
//
//		while (gpu_data->lane_pool.accumulated_offset[lane_id] >= gpu_data->lane_pool.lane_length[lane_id]) {
//			gpu_data->lane_pool.accumulated_offset[lane_id] -= gpu_data->lane_pool.speed_history[gpu_data->lane_pool.Tp[lane_id]][lane_id] * UNIT_TIME_STEPS;
//			gpu_data->lane_pool.Tp[lane_id] += UNIT_TIME_STEPS;
//		}
//	}
//}
//
//void supply_simulation_vehicle_passing(int time_step) {
//	for (unsigned int node_id = 0; node_id < the_network->all_nodes.size(); node_id++) {
//
//		for (int i = 0; i < gpu_data->node_pool.MAXIMUM_ACCUMULATED_FLOW[node_id]; i++) {
//			int lane_id = -1;
//
//			//Find A vehicle
//			GPUVehicle* one_v = get_next_vehicle_at_node(node_id, &lane_id);
//
//			if (one_v == NULL || lane_id < 0) {
////			printf("one_v == NULL\n");
//				break;
//			}
//
//			//Insert to next Lane
//			if (gpu_data->lane_pool.vehicle_space[0][lane_id]->next_path_index >= gpu_data->lane_pool.vehicle_space[0][lane_id]->whole_path_length) {
//				//the vehicle has finished the trip
//
////			printf("vehicle %d finish trip at node %d,\n", one_v->vehicle_ID, node_id);
//			}
//			else {
//				int next_lane_index = gpu_data->lane_pool.vehicle_space[0][lane_id]->path_code[gpu_data->lane_pool.vehicle_space[0][lane_id]->next_path_index];
//				int next_lane_id = gpu_data->node_pool.downstream[next_lane_index][node_id];
//				gpu_data->lane_pool.vehicle_space[0][lane_id]->next_path_index++;
//
//				//it is very critical to update the entry time when passing
//				gpu_data->lane_pool.vehicle_space[0][lane_id]->entry_time = time_step;
//
//				//add the vehicle
//				gpu_data->lane_pool.vehicle_passed_space[gpu_data->lane_pool.vehicle_passed_to_the_lane_counts[next_lane_id]][next_lane_id] = one_v;
//				gpu_data->lane_pool.vehicle_passed_to_the_lane_counts[next_lane_id]++;
//
//				gpu_data->lane_pool.input_capacity[next_lane_id]--;
//				gpu_data->lane_pool.predicted_empty_space[next_lane_id] -= VEHICLE_LENGTH;
//
////			printf("time_step=%d,one_v->vehicle_ID=%d,lane_id=%d, next_lane_id=%d, next_lane_index=%d\n", time_step, one_v->vehicle_ID, lane_id, next_lane_id, next_lane_index);
//			}
//
//			//Remove from current Lane
//			for (int j = 1; j < gpu_data->lane_pool.vehicle_counts[lane_id]; j++) {
//				gpu_data->lane_pool.vehicle_space[j - 1][lane_id] = gpu_data->lane_pool.vehicle_space[j][lane_id];
//			}
//
//			gpu_data->lane_pool.vehicle_counts[lane_id]--;
//			gpu_data->lane_pool.output_capacity[lane_id]--;
//			gpu_data->lane_pool.flow[lane_id]++;
//		}
//	}
//}
//
//GPUVehicle* get_next_vehicle_at_node(int node_id, int* lane_id) {
//
//	int maximum_waiting_time = -1;
////	int the_lane_id = -1;
//
//	for (int j = 0; j < MAX_LANE_UPSTREAM; j++) {
//
//		int one_lane_id = gpu_data->node_pool.upstream[j][node_id];
//		if (one_lane_id < 0) continue;
//
//		/*
//		 * Condition 1: The Lane is not NULL
//		 * ----      2: Has Output Capacity
//		 * ---       3: Is not blocked
//		 * ---       4: Has vehicles
//		 * ---       5: The vehicle can pass
//		 */
//
//		if (gpu_data->lane_pool.output_capacity[one_lane_id] > 0 && gpu_data->lane_pool.blocked[one_lane_id] == false && gpu_data->lane_pool.vehicle_counts[one_lane_id] > 0) {
//			int time_diff = gpu_data->lane_pool.Tp[one_lane_id] - gpu_data->lane_pool.vehicle_space[0][one_lane_id]->entry_time;
//			if (time_diff >= 0) {
//
//				//if already the final move, then no need for checking next road
//				if ((gpu_data->lane_pool.vehicle_space[0][one_lane_id]->next_path_index) >= (gpu_data->lane_pool.vehicle_space[0][one_lane_id]->whole_path_length)) {
//					if (time_diff > maximum_waiting_time) {
//						maximum_waiting_time = time_diff;
//						*lane_id = one_lane_id;
//						return gpu_data->lane_pool.vehicle_space[0][one_lane_id];
//					}
//				}
//				else {
//					int next_lane_index = gpu_data->lane_pool.vehicle_space[0][one_lane_id]->path_code[gpu_data->lane_pool.vehicle_space[0][one_lane_id]->next_path_index];
//					int next_lane_id = gpu_data->node_pool.downstream[next_lane_index][node_id];
//
//					/**
//					 * Condition 6: The Next Lane has input capacity
//					 * ---       7: The next lane has empty space
//					 */
//					if (gpu_data->lane_pool.input_capacity[next_lane_id] > 0 && gpu_data->lane_pool.predicted_empty_space[next_lane_id] > VEHICLE_LENGTH) {
//						if (time_diff > maximum_waiting_time) {
//							maximum_waiting_time = time_diff;
//							*lane_id = one_lane_id;
//							return gpu_data->lane_pool.vehicle_space[0][one_lane_id];
//						}
//					}
//					else {
//						gpu_data->lane_pool.blocked[one_lane_id] = true;
//					}
//				}
//			}
//		}
//	}
//
//	return NULL;
//}
//
//void supply_simulation_after_vehicle_passing(int time_step) {
//
//	for (unsigned int lane_id = 0; lane_id < the_network->all_links.size(); lane_id++) {
//
////update queue length
//		bool continue_loop = true;
//		float queue_length = 0;
//		float acc_length_moving = gpu_data->lane_pool.accumulated_offset[lane_id];
//		int to_time_step = gpu_data->lane_pool.Tp[lane_id];
//
//		for (int i = 0; continue_loop && i < gpu_data->lane_pool.vehicle_counts[lane_id]; i++) {
//			if (gpu_data->lane_pool.vehicle_space[i][lane_id]->entry_time <= gpu_data->lane_pool.Tp[lane_id]) {
//				queue_length += VEHICLE_LENGTH;
//			}
//			else {
//				int entry_time = gpu_data->lane_pool.vehicle_space[i][lane_id]->entry_time;
//				for (int j = entry_time; i < to_time_step; i++) {
//					acc_length_moving -= gpu_data->lane_pool.speed_history[j][lane_id] * UNIT_TIME_STEPS;
//				}
//
//				if (acc_length_moving + queue_length >= gpu_data->lane_pool.lane_length[lane_id]) {
//					to_time_step = entry_time;
//					queue_length += VEHICLE_LENGTH;
//				}
//				else {
//					continue_loop = false;
//				}
//			}
//		}
//
////update queue length
//		gpu_data->lane_pool.queue_length[lane_id] = queue_length;
//
////update the queue history
//		for (int i = 3; i > 0; i--) {
//			gpu_data->lane_pool.his_queue_length[i][lane_id] = gpu_data->lane_pool.his_queue_length[i - 1][lane_id];
//		}
//		gpu_data->lane_pool.his_queue_length[0][lane_id] = queue_length;
//
////update the empty space
//		if (gpu_data->lane_pool.new_vehicle_join_counts[lane_id] > 0) {
//			gpu_data->lane_pool.empty_space[lane_id] = gpu_data->lane_pool.speed[lane_id] * UNIT_TIME_STEPS - gpu_data->lane_pool.new_vehicle_join_counts[lane_id] * VEHICLE_LENGTH;
//			if (gpu_data->lane_pool.empty_space[lane_id] < 0) gpu_data->lane_pool.empty_space[lane_id] = 0;
//		}
//		else {
//			gpu_data->lane_pool.empty_space[lane_id] = gpu_data->lane_pool.empty_space[lane_id] + gpu_data->lane_pool.speed[lane_id] * UNIT_TIME_STEPS;
//		}
//	}
//}
