///**
// * This is the entry class to run ETS on CPU&GPU
// */
//
//#include "shared_include.h"
//#include "OnGPULanePool.h"
//#include "OnGPUVehiclePool.h"
//#include "OnGPUNewLaneVehicles.h"
//#include "OnGPUMemory.h"
//
//#include "OnGPUreadonly_include.h"
//#include "SimulationResults.h"
//
//#include "../network/Network.h"
//
//#include "../demand/OD_Pair.h"
//#include "../demand/OD_Path.h"
//#include "../demand/Vehicle.h"
//
//using namespace std;
//
//bool load_in_network();
//bool load_in_demand();
//bool initilizeCPU();
//bool initilizeGPU();
//bool initGPUData(GPUMemory* data_local);
//bool releaseAll();
//
////bool initilizeSimulationForFirstTimeStep();
//
///*
// * Time
// */
//long simulation_start_time;
//long simulation_end_time;
//long simulation_time_step;
//
///*
// * simulation_time is already finished time;
// * simulation_time + 1 might be the current simulating time on GPU
// */
//long simulated_time;
//
///*
// * simulation_results_outputed_time is already outputted time;
// * simulation_results_outputed_time + 1 might be the outputing time on CPU
// */
//long simulation_results_outputed_time;
//
///*
// * simulation_drivers_generation_time is already generated time;
// * simulation_drivers_generation_time + 1 might be the generating time on CPU
// */
////long simulation_drivers_generated_time;
////long simulation_infor_update_freq;
////long simulation_next_infor_update_time;
///*
// * Demand Part
// */
//bool generate_vehciles(int time_step);
//bool push_vehciles_to_GPU(int time_step);
//
///*
// * Supply Part
// */
//__global__ void supply_simulation_pre_vehicle_passing(GPUMemory* gpu_data, int time_step, int segment_length);
//__global__ void supply_simulation_vehicle_passing(GPUMemory* gpu_data, int time_step, int node_length);
//__global__ void supply_simulation_after_vehicle_passing(GPUMemory* gpu_data, int time_step, int segment_length);
//
//bool copy_simulated_results_to_CPU(int time_step);
//bool output_simulated_results(int time_step);
//bool update_travel_time(int time_step);
//
///*
// * GPU Streams
// * stream1: GPU Supply Simulation
// * stream2: Others on GPU
// */
//cudaStream_t stream1, stream2;
//cudaEvent_t GPU_supply_one_time_simulation_done_event;
//
///*
// * GPU threads settings
// */
//int roadBlocks;
//int nodeBlocks;
//int roadThreadsInABlock;
//int nodeThreadsInABlock;
//
///*
// *
// */
//Network* the_network;
//vector<OD_Pair*> all_od_pairs;
//vector<OD_Pair_PATH*> all_od_paths;
//vector<Vehicle*> all_vehicles;
//
///*
// *
// */
//std::string network_file_path = "data/network_10.dat";
//std::string demand_file_path = "data/demand_10.dat";
//std::string od_pair_file_path = "data/od_pair_10.dat";
//std::string od_pair_paths_file_path = "data/od_pair_paths_10.dat";
//
///*
// * data in GPU
// */
//GPUMemory* gpu_data;
//
///**
// * Simulation Results
// */
//std::map<int, SimulationResults*> simulation_results_pool;
//std::string simulation_output_file_path = "output/simulated_outputs.txt";
//ofstream simulation_results_output_file;
//
///**
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
//	//create streams
//	cudaStreamCreate(&stream1);
//	cudaStreamCreate(&stream2);
//
//	//create a event
//	cudaEventCreate(&GPU_supply_one_time_simulation_done_event);
//
////	if (initilizeSimulationForFirstTimeStep() == false) {
////		cout << "initilizeSimulationForFirstTimeStep fails" << endl;
////		return 0;
////	}
//
//	/*
//	 * Simulation Loop
//	 */
//
//	while (((simulated_time > simulation_end_time) && (simulation_results_outputed_time > simulation_end_time)) == false) {
//		if (simulated_time <= simulation_end_time && (cudaEventQuery(GPU_supply_one_time_simulation_done_event) == cudaSuccess)) {
//
//			//update simulated results
//			copy_simulated_results_to_CPU(simulated_time);
////			if (simulation_next_infor_update_time < simulated_time) {
////				update_travel_time(simulated_time);
////				simulation_next_infor_update_time += simulation_infor_update_freq;
////			}
//
////update time
//			simulated_time += simulation_time_step;
//			cout << "simulated_time:" << simulated_time << endl;
//
//			//can go to the next time step
////			if (simulation_drivers_generated_time <= simulated_time) {
////				// demand is not ready for the next time step
////				generate_vehciles(simulated_time + simulation_time_step);
////				push_vehciles_to_GPU(simulated_time + simulation_time_step);
////
////				simulation_drivers_generated_time += simulation_time_step;
////			}
//
////start GPU for supply part
//			supply_simulation_pre_vehicle_passing<<<roadBlocks, roadThreadsInABlock, 0, stream1>>>(gpu_data, simulated_time + simulation_time_step, LANE_SIZE);
//			supply_simulation_vehicle_passing<<<nodeBlocks, nodeThreadsInABlock, 0, stream1>>>(gpu_data, simulated_time + simulation_time_step, NODE_SIZE);
//			supply_simulation_after_vehicle_passing<<<roadBlocks, roadThreadsInABlock, 0, stream1>>>(gpu_data, simulated_time + simulation_time_step, LANE_SIZE);
//
//			cudaEventRecord(GPU_supply_one_time_simulation_done_event, stream1);
//		}
//		//If GPU is not finished, then try to generate vehicles on CPU
////		else if (simulation_drivers_generated_time <= simulation_next_infor_update_time) {
////			//generate vehicles
////
////			generate_vehciles(simulation_drivers_generated_time);
////			simulation_drivers_generated_time += simulation_time_step;
////		}
//		//If GPU is not finished and cannot generate vehicles on CPU, then try to output infor
//		else if (simulation_results_outputed_time < simulated_time) {
//			//generate vehicles
//
//			output_simulated_results(simulation_results_outputed_time);
//			simulation_results_outputed_time += simulation_time_step;
//		}
//		else {
//			cout << "---------------------" << endl;
//			cout << "CPU nothing to do" << endl;
//			cout << "simulated_time:" << simulated_time << endl;
////			cout << "simulation_drivers_generated_time:" << simulation_drivers_generated_time << endl;
//			cout << "simulation_results_outputed_time:" << simulation_results_outputed_time << endl;
//			cout << "---------------------" << endl;
//		}
//	}
//
//	cudaEventDestroy(GPU_supply_one_time_simulation_done_event);
//	cudaStreamDestroy(stream1);
//	cudaStreamDestroy(stream2);
//
//	return 1;
//}
//
///**
// *Init everything
// */
//
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
//	simulation_start_time = 0;
//	simulation_end_time = TOTAL_TIME_STEPS; // 2 hours
//	simulation_time_step = 1;
//
//	simulated_time = simulation_start_time;
//	simulation_results_outputed_time = simulation_start_time;
////	simulation_drivers_generated_time = simulation_start_time - simulation_time_step;
//
//	roadThreadsInABlock = 32;
//	nodeThreadsInABlock = 32;
//
//	roadBlocks = LANE_SIZE / roadThreadsInABlock + 1;
//	nodeBlocks = NODE_SIZE / nodeThreadsInABlock + 1;
//
//	simulation_results_pool.clear();
//
//	simulation_results_output_file.open(simulation_output_file_path.c_str());
//
//	//currently, not used.
////	simulation_infor_update_freq = 15 * 60;
////	simulation_next_infor_update_time = simulation_infor_update_freq;
//
//	return true;
//}
//
///*
// * This is a huge class to init GPU memory
// */
//bool initilizeGPU() {
//	gpu_data = NULL;
//
//	GPUMemory* data_local = new GPUMemory();
//	initGPUData(data_local);
//	data_local->test = 1;
//
//	if (cudaMalloc(&gpu_data, data_local->total_size()) != cudaSuccess) {
//		cerr << "cudaMalloc(&gpu_data, sizeof(GPUMemory)) failed" << endl;
//	}
//
//	cudaMemcpy(gpu_data, data_local, data_local->total_size(), cudaMemcpyHostToDevice);
//
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
//		assert(one_link->link_id == i);
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
//		data_local->lane_pool.lane_length[i] = 1000; // meter
//		data_local->lane_pool.max_vehicles[i] = 1000 / 5; //number of vehicles
//		data_local->lane_pool.output_capacity[i] = 1800 / 3600 * 2; //
//		data_local->lane_pool.input_capacity[i] = 1800 / 3600 * 2; //
//		data_local->lane_pool.empty_space[i] = 0;
//
//		/*
//		 * for speed calculation
//		 */
//		data_local->lane_pool.alpha[i] = 0.1;
//		data_local->lane_pool.beta[i] = 1;
//		data_local->lane_pool.max_density[i] = 200;
//		data_local->lane_pool.min_density[i] = 25;
//		data_local->lane_pool.MAX_SPEED[i] = 40;
//		data_local->lane_pool.MIN_SPEED[i] = 5;
//
//		/*
//		 * for connections
//		 * three possible values: 0, 1, 2
//		 */
//		data_local->lane_pool.to_node_id[i] = one_link->to_node->node_id;
////		int connected_lane_ID[MAX_LANE_CONNECTION][LANE_SIZE];
//
//		/*
//		 * for access vehicles, changed to array of pointers
//		 */
////		data_local->lane_pool.start_vehicle_pool_pos[i] = 200 * i;
////		data_local->lane_pool.last_vehicle_pos[i] = 200 * i;
////		data_local->lane_pool.end_vehicle_pool_pos[i] = 200 * (i + 1) - 1;
//		for (int j = 0; j < TOTAL_TIME_STEPS; j++) {
//			data_local->lane_pool.speed_history[j][i] = -1;
//		}
//
//		float weight[QUEUE_LENGTH_HISTORY];
//		weight[0] = 0.2;
//		weight[1] = 0.3;
//		weight[2] = 0.5;
//		weight[3] = 0;
//
////		{ 0.2, 0.3, 0.5, 0 };
//
//		for (int j = 0; j < QUEUE_LENGTH_HISTORY; j++) {
//			data_local->lane_pool.his_queue_length[j][i] = -1;
//			data_local->lane_pool.his_queue_length_weighting[j][i] = weight[j];
//		}
//
//		//inside vehicles
//		data_local->lane_pool.vehicle_counts[i] = 0;
//
//		for (int j = 0; j < MAX_VEHICLE_PER_LANE; j++) {
//			data_local->lane_pool.vehicle_space[j][i] = NULL;
//		}
//
//		//loading vehicles
//		data_local->lane_pool.vehicle_passed_to_the_lane[i] = 0;
//
//		for (int j = 0; j < LANE_INPUT_CAPACITY_TIME_STEP; j++) {
//			data_local->lane_pool.vehicle_passed_space[j][i] = NULL;
//		}
//
//		data_local->lane_pool.predicted_empty_space[i] = 0;
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
//	}
//
//	/**
//	 * Third Part:
//	 */
//
//	//Init NewLaneVehicles
////	for (int i = 0; i < TOTAL_TIME_STEPS; i++) {
////		for (int l = 0; l < LANE_SIZE; l++) {
////
////			data_local->new_vehicles_every_time_step[i]->new_vehicle_size[l] = 0;
////
////			for (int j = 0; j < VEHICLE_MAX_LOADING; j++) {
////				data_local.new_vehicles_every_time_step[i]->new_vehicles[j][l] = NULL;
////			}
////		}
////	}
//	//Init VehiclesSpaceInLane
////	for (int i = 0; i < LANE_SIZE; i++) {
////		for (int j = 0; j < MAX_VEHICLE_PER_LANE; j++) {
////			data_local->vehicle_in_lanes.vehicle_space[j][i] = NULL;
////		}
////	}
//	//Init VehiclePool
//	for (int i = 0; i < all_vehicles.size(); i++) {
//		Vehicle* one_vehicle = all_vehicles[i];
//		assert(one_vehicle->vehicle_id == i);
//
//		int time_index = one_vehicle->entry_time;
//		int lane_ID = all_od_paths[one_vehicle->path_id]->link_ids[0];
//
//		if (data_local->new_vehicles_every_time_step[time_index].new_vehicle_size[lane_ID] < VEHICLE_MAX_LOADING) {
//			int index = data_local->new_vehicles_every_time_step[time_index].new_vehicle_size[lane_ID];
//
//			data_local->new_vehicles_every_time_step[time_index].new_vehicles[index][lane_ID].entry_time = time_index;
//			data_local->new_vehicles_every_time_step[time_index].new_vehicles[index][lane_ID].current_lane_ID = lane_ID;
//
//			for (int p = 0; p < MAX_ROUTE_LENGTH; p++) {
//				data_local->new_vehicles_every_time_step[time_index].new_vehicles[index][lane_ID].path_code[p] = all_od_paths[one_vehicle->path_id]->route_code[p] ? 1 : 0;
//			}
//
////			data_local->new_vehicles_every_time_step[time_index].new_vehicles[index][lane_ID].path_code = all_od_paths[one_vehicle->path_id]->route_code;
//			data_local->new_vehicles_every_time_step[time_index].new_vehicles[index][lane_ID].next_path_index = 0;
//			data_local->new_vehicles_every_time_step[time_index].new_vehicles[index][lane_ID].whole_path_length = all_od_paths[one_vehicle->path_id]->link_ids.size();
//
//			data_local->new_vehicles_every_time_step[time_index].new_vehicle_size[lane_ID]++;
//		}
//	}
//
////	data_local->vehicle_in_lanes = new VehiclesSpaceInLane();
////	data_local->all_vehicles_in_GPU = new GPUVehicle();
//
//	return true;
//}
//
////bool initilizeSimulationForFirstTimeStep() {
////	generate_vehciles(simulation_start_time);
////	push_vehciles_to_GPU(simulation_start_time);
////
////	simulation_drivers_generated_time += simulation_time_step;
////
////	return true;
////}
//
//bool releaseAll() {
//
//	simulation_results_output_file.flush();
//	simulation_results_output_file.close();
//
//	return true;
//}
//
///**
// * Demand
// */
//bool generate_vehciles(int time_step) {
//	return true;
//}
//
//bool push_vehciles_to_GPU(int time_step) {
//	return true;
//}
//
///**
// * Supply
// */
//bool copy_simulated_results_to_CPU(int time_step) {
//	SimulationResults* one = new SimulationResults();
//
//	cudaMemcpy(one->flow, gpu_data->lane_pool.flow, sizeof(float) * LANE_SIZE, cudaMemcpyDeviceToHost);
//	cudaMemcpy(one->density, gpu_data->lane_pool.density, sizeof(float) * LANE_SIZE, cudaMemcpyDeviceToHost);
//	cudaMemcpy(one->speed, gpu_data->lane_pool.speed, sizeof(float) * LANE_SIZE, cudaMemcpyDeviceToHost);
//	cudaMemcpy(one->queue_length, gpu_data->lane_pool.queue_length, sizeof(float) * LANE_SIZE, cudaMemcpyDeviceToHost);
//
//	simulation_results_pool[time_step] = one;
//	return true;
//}
//
//bool output_simulated_results(int time_step) {
//
//	SimulationResults* one = simulation_results_pool[time_step];
//	assert(one != NULL);
//
////	simulation_results_output_file << time_step << ":" <<
//	for (int i = 0; i < LANE_SIZE; i++) {
//		simulation_results_output_file << time_step << ":" << one->flow[i] << ":" << one->density[i] << ":" << one->speed[i] << ":" << one->queue_length[i] << endl;
//	}
//
//	return true;
//}
//
//bool update_travel_time(int time_step) {
//	return true;
//}
//
///*
// * GPU functions
// */
//__global__ void supply_simulation_pre_vehicle_passing(GPUMemory* gpu_data, int time_step, int segment_length) {
//	int lane_id = blockIdx.x * blockDim.x + threadIdx.x;
//	if (lane_id >= segment_length) return;
//
////	std::cout << "lane_id:" << lane_id << std::endl;
//
//	//move new vehicles to buffer
////	for (int i = 0; i < gpu_data->new_vehicles_every_time_step[time_step]->new_vehicle_size[lane_id]; i++) {
////		//if has space
////		if (gpu_data->lane_pool->last_vehicle_pos[lane_id] <= gpu_data->lane_pool->end_vehicle_pool_pos[lane_id]) {
////			gpu_data->vehicle_in_lanes->vehicle_space[gpu_data->lane_pool->last_vehicle_pos[lane_id]][lane_id] = gpu_data->new_vehicles_every_time_step[time_step]->new_vehicles[i][lane_id];
////			gpu_data->lane_pool->last_vehicle_pos[lane_id]++;
////			gpu_data->lane_pool->vehicle_counts[lane_id]++;
////		}
////		else {
////			//is full, the vehicle is ignored.
////		}
////	}
//
//	/*
//	 * Init before starting
//	 */
//	gpu_data->lane_pool.input_capacity[lane_id] = LANE_INPUT_CAPACITY_TIME_STEP;
//	gpu_data->lane_pool.output_capacity[lane_id] = LANE_OUTPUT_CAPACITY_TIME_STEP;
//
//	//load passed vehicles to the back of the lane
//	//all passed vehicles must have enough space to go
//	for (int i = 0; i < gpu_data->lane_pool.vehicle_passed_to_the_lane[lane_id]; i++) {
//		if (gpu_data->lane_pool.vehicle_counts[lane_id] < gpu_data->lane_pool.max_vehicles[lane_id]) {
//			gpu_data->lane_pool.vehicle_space[gpu_data->lane_pool.vehicle_counts[lane_id]][lane_id] = gpu_data->lane_pool.vehicle_passed_space[i][lane_id];
//			gpu_data->lane_pool.vehicle_counts[lane_id]++;
//		}
//	}
//
//	gpu_data->lane_pool.vehicle_passed_to_the_lane[lane_id] = 0;
//
//	//load newly generated vehicles to the back of the lane
//	//if cannot load in the lane, vehicles will be ignored
//	for (int i = 0; i < gpu_data->new_vehicles_every_time_step[time_step].new_vehicle_size[lane_id]; i++) {
//		if (gpu_data->lane_pool.vehicle_counts[lane_id] < gpu_data->lane_pool.max_vehicles[lane_id]) {
//			gpu_data->lane_pool.vehicle_space[gpu_data->lane_pool.vehicle_counts[lane_id]][lane_id] = &(gpu_data->new_vehicles_every_time_step[time_step].new_vehicles[i][lane_id]);
//			gpu_data->lane_pool.vehicle_counts[lane_id]++;
//		}
//	}
//
//	//update speed and density
//	gpu_data->lane_pool.density[lane_id] = gpu_data->lane_pool.vehicle_counts[lane_id] / gpu_data->lane_pool.lane_length[lane_id];
//
//	//Linear Speed-Density Relationship
//	gpu_data->lane_pool.speed[lane_id] = gpu_data->lane_pool.MAX_SPEED[lane_id]
//			- gpu_data->lane_pool.MAX_SPEED[lane_id] * gpu_data->lane_pool.density[lane_id] / gpu_data->lane_pool.max_density[lane_id];
//	if (gpu_data->lane_pool.speed[lane_id] < gpu_data->lane_pool.MIN_SPEED[lane_id]) gpu_data->lane_pool.speed[lane_id] = gpu_data->lane_pool.MIN_SPEED[lane_id];
//
//	//update predicted empty_space
//	gpu_data->lane_pool.predicted_empty_space[lane_id] = 1;
//
////update speed history
//	gpu_data->lane_pool.speed_history[time_step][lane_id] = gpu_data->lane_pool.speed[lane_id];
//
////update tp
//	gpu_data->lane_pool.accumulated_offset[lane_id] += gpu_data->lane_pool.speed[lane_id] * UNIT_TIME_STEPS; //meter
//
//	while (gpu_data->lane_pool.accumulated_offset[lane_id] >= gpu_data->lane_pool.lane_length[lane_id]) {
//		gpu_data->lane_pool.accumulated_offset[lane_id] -= gpu_data->lane_pool.speed_history[gpu_data->lane_pool.Tp[lane_id]][lane_id] * UNIT_TIME_STEPS;
//		gpu_data->lane_pool.Tp[lane_id] += UNIT_TIME_STEPS;
//	}
//}
//
//__global__ void supply_simulation_vehicle_passing(GPUMemory* gpu_data, int time_step, int node_length) {
//	int node_id = blockIdx.x * blockDim.x + threadIdx.x;
//	if (node_id >= node_length) return;
//
//	for (int i = 0; i < gpu_data->node_pool.MAXIMUM_ACCUMULATED_FLOW[node_id]; i++) {
//		//get the next vehicle and pass to the next lane
//		GPUVehicle* one_v = NULL;
//		int maximum_waiting_time = 0;
//		int the_lane_id = -1;
//
//		//ordered vehicles
//		for (int j = 0; j < MAX_LANE_UPSTREAM; j++) {
//			int link_id = gpu_data->node_pool.upstream[j][i];
//			if (link_id >= 0) {
//				if (gpu_data->lane_pool.vehicle_counts[link_id] > 0 && gpu_data->lane_pool.output_capacity[link_id] > 0) {
//					if (gpu_data->lane_pool.vehicle_space[0][link_id]->entry_time <= gpu_data->lane_pool.Tp[link_id]) {
//						int waiting_time = (gpu_data->lane_pool.Tp[link_id] - gpu_data->lane_pool.vehicle_space[0][link_id]->entry_time);
//
//						if (waiting_time > maximum_waiting_time) {
//							//check the next lane's capacity and empty space
//							//this is specially for this case study
//							if (gpu_data->lane_pool.vehicle_space[0][the_lane_id]->next_path_index > gpu_data->lane_pool.vehicle_space[0][the_lane_id]->whole_path_length) {
//								//the vehicle has finished the trip
//								maximum_waiting_time = waiting_time;
//								one_v = gpu_data->lane_pool.vehicle_space[0][link_id];
//								the_lane_id = link_id;
//							}
//							else {
//								int next_index = gpu_data->lane_pool.vehicle_space[0][link_id]->next_path_index;
//								bool next_lane_bool = gpu_data->lane_pool.vehicle_space[0][link_id]->path_code[next_index];
//								int next_lane = next_lane_bool ? 1 : 0;
//
//								int next_lane_id = gpu_data->node_pool.downstream[next_lane][i];
//								if (gpu_data->lane_pool.input_capacity[next_lane_id] > 0 && gpu_data->lane_pool.predicted_empty_space[next_lane_id] >= VEHICLE_MAX_LOADING) {
//									//can pass but not really pass
//									maximum_waiting_time = waiting_time;
//									one_v = gpu_data->lane_pool.vehicle_space[0][link_id];
//									the_lane_id = link_id;
//								}
//							}
//						}
//					}
//				}
//			}
//		}
//
//		//no vehcile can pass
//		if (one_v == NULL) {
////			std::cout << "no vehicle" << std::endl;
//			return;
//		}
//		else {
//			//remove the vehicle
//			for (int i = 1; i < gpu_data->lane_pool.vehicle_counts[the_lane_id]; i++) {
//				gpu_data->lane_pool.vehicle_space[i - 1][the_lane_id] = gpu_data->lane_pool.vehicle_space[i][the_lane_id];
//			}
//			gpu_data->lane_pool.vehicle_counts[the_lane_id]--;
//			gpu_data->lane_pool.output_capacity[the_lane_id]--;
//
//			gpu_data->lane_pool.flow[the_lane_id]++;
//
//			//if the vehicle pass the end
//			if (gpu_data->lane_pool.vehicle_space[0][the_lane_id]->next_path_index > gpu_data->lane_pool.vehicle_space[0][the_lane_id]->whole_path_length) {
//				//the vehicle has finished the trip
//				return;
//			}
//
//			bool next_lane_bool = gpu_data->lane_pool.vehicle_space[0][the_lane_id]->path_code[gpu_data->lane_pool.vehicle_space[0][the_lane_id]->next_path_index];
//			int next_lane = next_lane_bool ? 1 : 0;
//			int next_lane_id = gpu_data->node_pool.downstream[next_lane][i];
//
//			//add the vehicle
//			gpu_data->lane_pool.vehicle_passed_space[gpu_data->lane_pool.vehicle_counts[next_lane]][next_lane_id] = one_v;
//			gpu_data->lane_pool.vehicle_counts[next_lane_id]++;
//			gpu_data->lane_pool.input_capacity[next_lane_id]--;
//		}
//		//
//	}
//}
//
//__global__ void supply_simulation_after_vehicle_passing(GPUMemory* gpu_data, int time_step, int segment_length) {
//	int lane_id = blockIdx.x * blockDim.x + threadIdx.x;
//	if (lane_id >= segment_length) return;
//
//	//update queue length and the empty space
//	gpu_data->lane_pool.empty_space[lane_id] = 1;
//}
////
