///**
// *This project targets to check GPU is an option for DynaMIT.
// *This project also targets for a paper "Mesoscopic Traffic Simulation on GPU"
// */
//
//#include "../components_on_cpu/network/network.h"
//#include "../components_on_cpu/demand/od_pair.h"
//#include "../components_on_cpu/demand/od_path.h"
//#include "../components_on_cpu/demand/vehicle.h"
//#include "../components_on_cpu/util/time_tools.h"
//#include "../components_on_cpu/util/string_tools.h"
//#include "../components_on_cpu/util/simulation_results.h"
//#include "../components_on_cpu/util/shared_cpu_include.h"
//
//#include "../components_on_gpu/supply/on_GPU_memory.h"
//#include "../components_on_gpu/supply/on_GPU_vehicle.h"
//#include "../components_on_gpu/supply/on_GPU_new_lane_vehicles.h"
//#include "../components_on_gpu/util/shared_gpu_include.h"
//#include "../components_on_gpu/util/on_gpu_configuration.h"
//#include "../components_on_gpu/on_GPU_kernal.cuh"
//#include "../components_on_gpu/on_GPU_Macro.h"
//
//#include <cmath>
//
//using namespace std;
//
///**
// * CUDA Execution Configuration
// */
//int road_blocks;
//const int road_threads_in_a_block = 192;
//int node_blocks;
//const int node_threads_in_a_block = 192;
//
///*
// * Demand
// */
//Network* the_network;
//std::vector<ODPair*> all_od_pairs;
//std::vector<ODPairPATH*> all_od_paths;
//std::vector<Vehicle*> all_vehicles;
//
///*
// * Path Input Config
// */
//std::string network_file_path = "data_inputs/sg_network/network.dat";
//std::string demand_file_path = "data_inputs/sg_network/demand_106386.dat";
//std::string od_pair_file_path = "data_inputs/sg_network/od_1428.dat";
//std::string od_pair_paths_file_path = "data_inputs/sg_network/path_3349.dat";
//
///*
// * All data in GPU
// */
//GPUMemory* gpu_data;
////GPUSharedParameter* parameter_seeting_on_gpu;
//#ifdef ENABLE_CONSTANT_MEMORY
//__constant__ GPUSharedParameter data_setting_gpu_constant;
//#endif
//
////A large memory space is pre-defined in order to copy to GPU
//GPUVehicle *vpool_cpu;
////GPUVehicle *vpool_gpu;
//
////int *vpool_cpu_index;
////int *vpool_gpu_index;
//
///**
// * Simulation Results
// */
//std::string simulation_output_file_path = "output/test3.txt";
//std::map<int, SimulationResults*> simulation_results_pool;
//ofstream simulation_results_output_file;
//
////buffer is only used when kGPUToCPUSimulationResultsCopyBufferSize > 1
//SimulationResults* simulation_results_buffer_on_gpu;
//
////Used for buffer at CPU side
//SimulationResults* one_buffer = NULL;
//
///*
// * GPU Streams
// * stream1: GPU Supply Simulation
// */
//cudaStream_t stream_gpu_supply;
//cudaStream_t stream_gpu_io;
//cudaEvent_t gpu_supply_one_tick_simulation_done_trigger_event;
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
// */
////std::map<int, int> link_ID_to_link_Index;
////std::map<int, int> link_Index_to_link_ID;
////std::map<int, int> node_ID_to_node_Index;
////std::map<int, int> node_index_to_node_ID;
///*
// * Define Major Functions
// */
//bool InitParams(int argc, char* argv[]);
//bool LoadInNetwork();
//bool LoadInDemand();
//bool InitilizeCPU();
//bool initilizeGPU();
//
////bool InitGPUParameterSetting(GPUSharedParameter* data_setting_gpu);
//bool initGPUData(GPUMemory* data_local);
//bool StartSimulation();
//bool DestoryResources();
//
///*
// * Define Helper Functions
// */
//StringTools* str_tools;
//bool CopySimulatedResultsToCPU(int time_step);
//bool CopyBufferSimulatedResultsToCPU(int time_step);
//bool OutputSimulatedResults(int time_step);
//bool OutputBufferedSimulatedResults(int time_step);
//
//inline int TimestepToArrayIndex(int time_step) {
//	return (time_step - kStartTimeSteps) / kUnitTimeStep;
//}
//
///*
// * Supply Function Define
// */
//void supply_simulation_pre_vehicle_passing(int time_step);
//void supply_simulation_vehicle_passing(int time_step);
//void supply_simulation_after_vehicle_passing(int time_step);
//int GetNextVehicleAtNode(int node_id, int* lane_id);
//
///*
// * MAIN
// */
//int main(int argc, char* argv[]) {
//
//	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
//
//	cout << "Program Starts" << endl;
//
//	if (InitParams(argc, argv) == false) {
//		cout << "InitParams fails" << endl;
//		return 0;
//	}
//
//	if (LoadInNetwork() == false) {
//		cout << "Loading network fails" << endl;
//		return 0;
//	}
//
//	if (LoadInDemand() == false) {
//		cout << "Loading demand fails" << endl;
//		return 0;
//	}
//
//	if (InitilizeCPU() == false) {
//		cout << "InitilizeCPU fails" << endl;
//		return 0;
//	}
//
//	if (initilizeGPU() == false) {
//		cout << "initilizeGPU fails" << endl;
//		return 0;
//	}
//
//	//create streams
//	cudaStreamCreate(&stream_gpu_supply);
//	cudaStreamCreate(&stream_gpu_io);
//
//	//create a event
//	cudaEventCreate(&gpu_supply_one_tick_simulation_done_trigger_event);
//
//	std::cout << "Simulation Starts" << std::endl;
//
//	TimeTools profile;
//	profile.start_profiling();
//
//	//Start Simulation (ETSF implemented inside)
//	if (StartSimulation() == false) {
//		cout << "Simulation Fails" << endl;
//		DestoryResources();
//		return 0;
//	}
//
//	profile.end_profiling();
//	profile.output();
//
//	DestoryResources();
//	cout << "Simulation Succeed!" << endl;
//
//#ifdef _WIN32
//	system("pause");
//#endif
//
//	return 0;
//}
//
///**
// *
// */
//bool InitParams(int argc, char* argv[]) {
//	if (argc == 5) {
//		network_file_path = argv[2];
//		demand_file_path = argv[3];
//		simulation_output_file_path = argv[4];
//		std::cout << "parameters updated" << std::endl;
//	}
//	return true;
//}
//bool LoadInNetwork() {
//	the_network = new Network();
//	return Network::load_network(*the_network, network_file_path);
//}
//
//bool LoadInDemand() {
//	if (ODPair::load_in_all_ODs(all_od_pairs, od_pair_file_path) == false) {
//		return false;
//	}
//	if (ODPairPATH::load_in_all_OD_Paths(all_od_paths, od_pair_paths_file_path) == false) {
//		return false;
//	}
//	if (Vehicle::load_in_all_vehicles(all_vehicles, demand_file_path) == false) {
//		return false;
//	}
//	return true;
//}
//
//bool InitilizeCPU() {
//	simulation_start_time = kStartTimeSteps;
//	simulation_end_time = kEndTimeSteps; // 1 hour
//	simulation_time_step = kUnitTimeStep;
//
//	assert(simulation_time_step == 1);
//
//	to_simulate_time = simulation_start_time;
//	to_output_simulation_result_time = simulation_start_time;
//
//	road_blocks = kLaneSize / road_threads_in_a_block + 1;
//	node_blocks = kNodeSize / node_threads_in_a_block + 1;
//
//	simulation_results_pool.clear();
//	simulation_results_output_file.open(simulation_output_file_path.c_str());
//	simulation_results_output_file << "##TIME STEP" << ":Lane ID:" << ":(" << "COUNTS" << ":" << "flow" << ":" << "density" << ":" << "speed" << ":" << "queue_length" << ")" << endl;
//	str_tools = new StringTools();
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
// * Build a GPU data from the network data
// */
//bool initGPUData(GPUMemory* data_local) {
//
//	/**
//	 * First Part: Lane
//	 */
//
//	for (int i = 0; i < the_network->link_size; i++) {
//		Link* one_link = the_network->all_links[i];
//
//		//
//		assert(one_link->link_id == i);
//		data_local->lane_pool.lane_ID[i] = one_link->link_id;
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
//		data_local->lane_pool.lane_length[i] = one_link->length; // meter
//		data_local->lane_pool.max_vehicles[i] = (one_link->vehicle_end - one_link->vehicle_start + 1); //number of vehicles
//		data_local->lane_pool.output_capacity[i] = kLaneOutputCapacityPerTimeStep; //
//		data_local->lane_pool.input_capacity[i] = kLaneInputCapacityPerTimeStep; //
//		data_local->lane_pool.empty_space[i] = one_link->length;
//
//		/*
//		 * for speed calculation
//		 */
//		data_local->lane_pool.alpha[i] = kAlpha;
//		data_local->lane_pool.beta[i] = kBeta;
//		data_local->lane_pool.max_density[i] = kMaxDensity;
//		data_local->lane_pool.min_density[i] = kMinDensity;
//		data_local->lane_pool.MAX_speed[i] = kMaxSpeed;
//		data_local->lane_pool.MIN_speed[i] = kMinSpeed;
//
//		data_local->lane_pool.vehicle_counts[i] = 0;
//		data_local->lane_pool.vehicle_passed_to_the_lane_counts[i] = 0;
//
//		data_local->lane_pool.vehicle_start_index[i] = one_link->vehicle_start;
//		data_local->lane_pool.vehicle_end_index[i] = one_link->vehicle_end;
//
//		data_local->lane_pool.buffered_vehicle_start_index[i] = one_link->buffered_vehicle_start;
//		data_local->lane_pool.buffered_vehicle_end_index[i] = one_link->buffered_vehicle_end;
//
//		for (int j = 0; j < kTotalTimeSteps; j++) {
//			data_local->lane_pool.speed_history[j][i] = -1;
//		}
//
//		//it is assumed that QUEUE_LENGTH_HISTORY = 4;
//		assert(kQueueLengthHistory == 4);
//		float weight[kQueueLengthHistory];
//		weight[0] = 1.0;
//		weight[1] = 0;
//		weight[2] = 0;
//		weight[3] = 0;
//
//		//		{ 0.2, 0.3, 0.5, 0 };
//
//		for (int j = 0; j < kQueueLengthHistory; j++) {
//			data_local->lane_pool.his_queue_length[j][i] = -1;
//			data_local->lane_pool.his_queue_length_weighting[j][i] = weight[j];
//		}
//
//		data_local->lane_pool.predicted_empty_space[i] = 0;
//		data_local->lane_pool.predicted_queue_length[i] = 0;
//		data_local->lane_pool.last_time_empty_space[i] = 0;
//	}
//
//	/**
//	 * Second Part: Node
//	 */
////	NodePool* the_node_pool = data_local->node_pool;
//	for (int i = 0; i < the_network->node_size; i++) {
//		Node* one_node = the_network->all_nodes[i];
//
//		data_local->node_pool.node_ID[i] = one_node->node_id;
//		data_local->node_pool.max_acc_flow[i] = 0;
//		data_local->node_pool.acc_upstream_capacity[i] = 0;
//		data_local->node_pool.acc_downstream_capacity[i] = 0;
//
//		data_local->node_pool.upstream_lane_start_index[i] = one_node->up_lane_start_index;
//		data_local->node_pool.upstream_lane_end_index[i] = one_node->up_lane_end_index;
//
//		for (int j = one_node->up_lane_start_index; j <= one_node->up_lane_end_index; j++) {
//			data_local->node_pool.acc_upstream_capacity[i] += kLaneInputCapacityPerTimeStep;
//		}
//
//		data_local->node_pool.max_acc_flow[i] = data_local->node_pool.acc_upstream_capacity[i];
//	}
//
//	/**
//	 * Third Part:
//	 */
//
////Init VehiclePool
//	for (int i = kStartTimeSteps; i < kEndTimeSteps; i += kUnitTimeStep) {
//		for (int j = 0; j < kLaneSize; j++) {
//			data_local->new_vehicles_every_time_step[i].new_vehicle_size[j] = 0;
//			data_local->new_vehicles_every_time_step[i].lane_ID[j] = -1;
//		}
//	}
//
//	std::cout << "all_vehicles.size():" << all_vehicles.size() << std::endl;
//
////init host vehicle pool data /*xiaosong*/
//	int memory_space_for_vehicles = all_vehicles.size() * sizeof(GPUVehicle);
//	vpool_cpu = (GPUVehicle*) malloc(memory_space_for_vehicles);
//	if (vpool_cpu == NULL)
//		exit(1);
//
//	for (int i = kStartTimeSteps; i < kEndTimeSteps; i += kUnitTimeStep) {
//		for (int j = 0; j < kLaneSize; j++) {
//			for (int z = 0; z < kLaneInputCapacityPerTimeStep; z++) {
//				//init as no vehicle
//				data_local->new_vehicles_every_time_step[i].new_vehicles[j][z] = -1;
//			}
//		}
//	}
//
////	int nVehiclePerTick = kLaneInputCapacityPerTimeStep * kLaneSize;
////	std::cout << "init all_vehicles" << std::endl;
//
//	int total_inserted_vehicles = 0;
//
////Insert Vehicles
//	for (int i = 0; i < all_vehicles.size(); i++) {
//		Vehicle* one_vehicle = all_vehicles[i];
//
//		int time_index = one_vehicle->entry_time;
//		int time_index_covert = TimestepToArrayIndex(time_index);
//		assert(time_index == time_index_covert);
//
//		//try to load vehicles beyond the simulation border
//		if (time_index_covert >= kTotalTimeSteps)
//			continue;
//
//		int lane_ID = all_od_paths[one_vehicle->path_id]->link_ids[0];
//		int lane_Index = lane_ID; //the same for the SG Expressway case
//
//		if (data_local->new_vehicles_every_time_step[time_index_covert].new_vehicle_size[lane_Index] < kLaneInputCapacityPerTimeStep) {
//			int last_vehicle_index = data_local->new_vehicles_every_time_step[time_index_covert].new_vehicle_size[lane_Index];
//
//			vpool_cpu[i].vehicle_ID = one_vehicle->vehicle_id;
//			vpool_cpu[i].entry_time = time_index;
//			vpool_cpu[i].current_lane_ID = lane_Index;
//
//			assert(kMaxRouteLength > all_od_paths[one_vehicle->path_id]->link_ids.size());
//			int max_copy_length = kMaxRouteLength > all_od_paths[one_vehicle->path_id]->link_ids.size() ? all_od_paths[one_vehicle->path_id]->link_ids.size() : kMaxRouteLength;
//
//			for (int p = 0; p < max_copy_length; p++) {
//				vpool_cpu[i].path_code[p] = all_od_paths[one_vehicle->path_id]->link_ids[p];
//			}
//
//			//ready for the next lane, so next_path_index is set to 1, if the next_path_index == whole_path_length, it means cannot find path any more, can exit;
//			vpool_cpu[i].next_path_index = 1;
//			vpool_cpu[i].whole_path_length = all_od_paths[one_vehicle->path_id]->link_ids.size();
//
//			//will be re-writen by GPU
//			data_local->new_vehicles_every_time_step[time_index_covert].new_vehicles[lane_Index][last_vehicle_index] = vpool_cpu[i].vehicle_ID;
//			data_local->new_vehicles_every_time_step[time_index_covert].new_vehicle_size[lane_Index]++;
//
//			total_inserted_vehicles++;
//		} else {
////			std::cout << "Loading Vehicles Exceeds The Loading Capacity: Time:" << time_index_covert << ", Lane_ID:" << lane_ID << ",i:" << i << ",ID:" << one_vehicle->vehicle_id << std::endl;
//		}
//	}
//
//	std::cout << "init all_vehicles:" << total_inserted_vehicles << std::endl;
//
//	return true;
//}
//
//bool DestoryResources() {
//	simulation_results_output_file.flush();
//	simulation_results_output_file.close();
//
//	if (vpool_cpu != NULL)
//		delete vpool_cpu;
//	if (str_tools != NULL)
//		delete str_tools;
//
//	cudaDeviceReset();
//	return true;
//}
//
//bool StartSimulation() {
//	while (((to_simulate_time >= simulation_end_time) && (to_output_simulation_result_time >= simulation_end_time)) == false) {
//
//#ifdef ENABLE_OUTPUT
////		std::cout << "Current Time: " << to_simulate_time << std::endl;
//#endif
//
//		supply_simulation_pre_vehicle_passing(to_simulate_time);
//
//		supply_simulation_vehicle_passing(to_simulate_time);
//
//		supply_simulation_after_vehicle_passing(to_simulate_time);
//
//#ifdef ENABLE_OUTPUT
//		OutputSimulatedResults(to_output_simulation_result_time);
//#endif
//
//		to_simulate_time += simulation_time_step;
//		to_output_simulation_result_time += simulation_time_step;
//	}
//
//	return true;
//}
//
//bool OutputSimulatedResults(int time_step) {
//	if (time_step % 60 != 0)
//		return true;
//
//	for (int i = 0; i < kLaneSize; i++) {
//		int lane_ID = i;
//		int lane_Index = lane_ID;
//
//		simulation_results_output_file << time_step << ":lane:" << lane_ID << ":(" << gpu_data->lane_pool.vehicle_counts[lane_Index] << ":" << gpu_data->lane_pool.flow[lane_Index] << ":"
//				<< gpu_data->lane_pool.density[lane_Index] << ":" << gpu_data->lane_pool.speed[lane_Index] << ":" << gpu_data->lane_pool.queue_length[lane_Index] << ")" << endl;
//	}
//
//	return true;
//}
//
///**
// * Minor Functions
// */
//bool CopySimulatedResultsToCPU(int time_step) {
//	int index = TimestepToArrayIndex(time_step);
//	SimulationResults* one = new SimulationResults();
//
//	cudaMemcpy(one->flow, gpu_data->lane_pool.flow, sizeof(float) * kLaneSize, cudaMemcpyDeviceToHost);
//	cudaMemcpy(one->density, gpu_data->lane_pool.density, sizeof(float) * kLaneSize, cudaMemcpyDeviceToHost);
//	cudaMemcpy(one->speed, gpu_data->lane_pool.speed, sizeof(float) * kLaneSize, cudaMemcpyDeviceToHost);
//	cudaMemcpy(one->queue_length, gpu_data->lane_pool.queue_length, sizeof(float) * kLaneSize, cudaMemcpyDeviceToHost);
//	cudaMemcpy(one->counts, gpu_data->lane_pool.vehicle_counts, sizeof(int) * kLaneSize, cudaMemcpyDeviceToHost);
//	simulation_results_pool[index] = one;
//
//	return true;
//}
//
//bool CopyBufferSimulatedResultsToCPU(int time_step) {
//	cudaMemcpyAsync(one_buffer, simulation_results_buffer_on_gpu, sizeof(SimulationResults) * kGPUToCPUSimulationResultsCopyBufferSize, cudaMemcpyDeviceToHost, stream_gpu_io);
//
//	for (int i = 0; i < kGPUToCPUSimulationResultsCopyBufferSize; i++) {
//		int time_index = time_step - (kGPUToCPUSimulationResultsCopyBufferSize - 1) + i;
//		simulation_results_pool[time_index] = &one_buffer[i];
//	}
//
//	return true;
//}
//
//bool OutputBufferedSimulatedResults(int time_step) {
//	std::cout << "OutputBufferedSimulatedResults AT time " << time_step << std::endl;
//
//	for (int i = 0; i < kGPUToCPUSimulationResultsCopyBufferSize; i++) {
//		OutputSimulatedResults(time_step + i);
//	}
//
//	return true;
//}
//
///*
// * Supply Function Implementation
// */
//
//void supply_simulation_pre_vehicle_passing(int time_step) {
//	int time_index = time_step;
//
//	for (unsigned int lane_index = 0; lane_index < the_network->link_size; lane_index++) {
//
//		gpu_data->lane_pool.input_capacity[lane_index] = kLaneInputCapacityPerTimeStep;
//		gpu_data->lane_pool.output_capacity[lane_index] = kLaneOutputCapacityPerTimeStep;
//
//		//init for next GPU kernel function
//		gpu_data->lane_pool.blocked[lane_index] = false;
//
//		//load passed vehicles to the back of the lane
//		for (int i = 0; i < gpu_data->lane_pool.vehicle_passed_to_the_lane_counts[lane_index]; i++) {
//			if (gpu_data->lane_pool.vehicle_counts[lane_index] < gpu_data->lane_pool.max_vehicles[lane_index]) {
//				int new_vehicle_index = gpu_data->lane_pool.vehicle_counts[lane_index] + gpu_data->lane_pool.vehicle_start_index[lane_index];
//				int new_buffer_vehicle_index = i + gpu_data->lane_pool.buffered_vehicle_start_index[lane_index];
//
//				//pass the vehicle
//				gpu_data->lane_vehicle_pool.vehicle_space[new_vehicle_index] = gpu_data->lane_buffered_vehicle_pool.buffered_vehicle_space[new_buffer_vehicle_index];
//				gpu_data->lane_pool.vehicle_counts[lane_index]++;
//
//				//gpu_data->lane_pool.new_vehicle_join_counts[lane_index]++;
//			}
//		}
//
//		if (gpu_data->lane_pool.vehicle_passed_to_the_lane_counts[lane_index] > 0) {
//			gpu_data->lane_pool.empty_space[lane_index] = std::min(gpu_data->lane_pool.speed[lane_index] * kUnitTimeStep, gpu_data->lane_pool.empty_space[lane_index])
//					- gpu_data->lane_pool.vehicle_passed_to_the_lane_counts[lane_index] * kVehicleLength;
//
//			if (gpu_data->lane_pool.empty_space[lane_index] < 0)
//				gpu_data->lane_pool.empty_space[lane_index] = 0;
//		}
//
//		gpu_data->lane_pool.last_time_empty_space[lane_index] = gpu_data->lane_pool.empty_space[lane_index];
//		gpu_data->lane_pool.vehicle_passed_to_the_lane_counts[lane_index] = 0;
//
//		//
//		//load newly generated vehicles to the back of the lane
//		for (int i = 0; i < gpu_data->new_vehicles_every_time_step[time_index].new_vehicle_size[lane_index]; i++) {
//			if (gpu_data->lane_pool.vehicle_counts[lane_index] < gpu_data->lane_pool.max_vehicles[lane_index]) {
//				int new_vehicle_index = gpu_data->lane_pool.vehicle_counts[lane_index] + gpu_data->lane_pool.vehicle_start_index[lane_index];
//
//				gpu_data->lane_vehicle_pool.vehicle_space[new_vehicle_index] = (gpu_data->new_vehicles_every_time_step[time_index].new_vehicles[lane_index][i]);
//				gpu_data->lane_pool.vehicle_counts[lane_index]++;
//			}
//		}
//
//		float density_ = 0.0f;
//		float speed_ = 0.0f;
//
//		//update speed and density
//		density_ = 1.0 * kVehicleLength * gpu_data->lane_pool.vehicle_counts[lane_index] / gpu_data->lane_pool.lane_length[lane_index];
//
//		if (density_ < kMinDensity)
//			speed_ = kMaxSpeed;
//		else {
//			speed_ = kMaxSpeed * pow(1.0 - pow((density_ - kMinDensity) / kMaxDensity, kBeta), kAlpha);
//
////			speed_ = kMaxSpeed - kMaxSpeed / (kMaxDensity - kMinDensity) * (density_ - kMinDensity);
//		}
//		//		gpu_data->lane_pool.speed[lane_index] = ( gpu_data->lane_pool.MAX_SPEED[lane_index] - gpu_data->lane_pool.MIN_SPEED ) / gpu_data->lane_pool.max_density[lane_index] * ( gpu_data->lane_pool.max_density[lane_index] - 0 );
//
//		if (speed_ < kMinSpeed)
//			speed_ = kMinSpeed;
//
//		//update speed history
//		gpu_data->lane_pool.speed_history[time_index][lane_index] = speed_;
//
//		gpu_data->lane_pool.density[lane_index] = density_;
//		gpu_data->lane_pool.speed[lane_index] = speed_;
//		//estimated empty_space
//
//		float prediction_queue_length_ = 0.0f;
//
//		if (time_step < kStartTimeSteps + 4 * kUnitTimeStep) {
//			//		gpu_data->lane_pool.predicted_empty_space[lane_index] = gpu_data->lane_pool.his_queue_length[0][lane_index];
//			//		gpu_data->lane_pool.predicted_queue_length[lane_index] = 0;
//
//			gpu_data->lane_pool.predicted_empty_space[lane_index] = std::min(gpu_data->lane_pool.last_time_empty_space[lane_index] + (gpu_data->lane_pool.speed[lane_index] * kUnitTimeStep),
//					1.0f * gpu_data->lane_pool.lane_length[lane_index]);
//		} else {
//			prediction_queue_length_ = gpu_data->lane_pool.his_queue_length[0][lane_index];
//			prediction_queue_length_ += (gpu_data->lane_pool.his_queue_length[0][lane_index] - gpu_data->lane_pool.his_queue_length[1][lane_index])
//					* gpu_data->lane_pool.his_queue_length_weighting[0][lane_index];
//
//			//		prediction_empty_space_ += (gpu_data->lane_pool.his_queue_length[1][lane_index] - gpu_data->lane_pool.his_queue_length[2][lane_index])
//			//				* gpu_data->lane_pool.his_queue_length_weighting[1][lane_index];
//			//
//			//		prediction_empty_space_ += (gpu_data->lane_pool.his_queue_length[2][lane_index] - gpu_data->lane_pool.his_queue_length[3][lane_index])
//			//				* gpu_data->lane_pool.his_queue_length_weighting[2][lane_index];
//
//			gpu_data->lane_pool.predicted_empty_space[lane_index] = std::min(gpu_data->lane_pool.last_time_empty_space[lane_index] + (gpu_data->lane_pool.speed[lane_index] * kUnitTimeStep),
//					(gpu_data->lane_pool.lane_length[lane_index] - prediction_queue_length_));
//		}
//
//		//	gpu_data->lane_pool.debug_data[lane_index] = gpu_data->lane_pool.predicted_empty_space[lane_index];
//		//update Tp
//
//		gpu_data->lane_pool.accumulated_offset[lane_index] += gpu_data->lane_pool.speed[lane_index] * kUnitTimeStep; //meter
//
//		while (gpu_data->lane_pool.accumulated_offset[lane_index] >= gpu_data->lane_pool.lane_length[lane_index]) {
//			gpu_data->lane_pool.accumulated_offset[lane_index] -= gpu_data->lane_pool.speed_history[gpu_data->lane_pool.Tp[lane_index]][lane_index] * kUnitTimeStep;
//			gpu_data->lane_pool.Tp[lane_index] += kUnitTimeStep;
//		}
//
//		//update queue length
//		int queue_start = gpu_data->lane_pool.queue_length[lane_index] / kVehicleLength;
//		for (int queue_index = queue_start; queue_index < gpu_data->lane_pool.vehicle_counts[lane_index]; queue_index++) {
//			int vehicle_index = gpu_data->lane_vehicle_pool.vehicle_space[gpu_data->lane_pool.vehicle_start_index[lane_index] + queue_index];
//			//		if (gpu_data->lane_pool.vehicle_space[queue_index][lane_index]->entry_time <= gpu_data->lane_pool.Tp[lane_index]) {
//			if (vpool_cpu[vehicle_index].entry_time <= gpu_data->lane_pool.Tp[lane_index]) {
//				gpu_data->lane_pool.queue_length[lane_index] += kVehicleLength;
//			} else {
//				break;
//			}
//		}
//	}
//}
//
//int GetNextVehicleAtNode(int node_index, int* lane_index) {
//
//	int maximum_waiting_time = -1;
//	int the_one_veh = -1;
//
//	int upstream_start_lane = gpu_data->node_pool.upstream_lane_start_index[node_index];
//	int upstream_end_lane = gpu_data->node_pool.upstream_lane_end_index[node_index];
//
//	//no upstream links, so, return -1, no vehicle
//	if (upstream_start_lane < 0 || upstream_end_lane < 0)
//		return -1;
//
//	for (int one_lane_index = upstream_start_lane; one_lane_index <= upstream_end_lane; one_lane_index++) {
//		/*
//		 * Condition 1: The Lane is not NULL
//		 * ----      2: Has Output Capacity
//		 * ---       3: Is not blocked
//		 * ---       4: Has vehicles
//		 * ---       5: The vehicle can pass
//		 */
//
//		if (gpu_data->lane_pool.output_capacity[one_lane_index] > 0 && gpu_data->lane_pool.blocked[one_lane_index] == false && gpu_data->lane_pool.vehicle_counts[one_lane_index] > 0) {
////			int start_vehicle_index = gpu_data->lane_pool.vehicle_start_index;
////			int end_vehicle_index = gpu_data->lane_pool.vehicle_end_index;
//
//			int first_vehicle_ID = gpu_data->lane_vehicle_pool.vehicle_space[gpu_data->lane_pool.vehicle_start_index[one_lane_index]];
//
////			int time_diff = gpu_data->lane_pool.Tp[one_lane_index] - gpu_data->lane_pool.vehicle_space[0][one_lane_index]->entry_time;
//			int time_diff = gpu_data->lane_pool.Tp[one_lane_index] - vpool_cpu[first_vehicle_ID].entry_time;
//			if (time_diff >= 0) {
//
//				//if already the final move, then no need for checking next road
//				if ((vpool_cpu[first_vehicle_ID].next_path_index) >= (vpool_cpu[first_vehicle_ID].whole_path_length)) {
//					if (time_diff > maximum_waiting_time) {
//						maximum_waiting_time = time_diff;
//						*lane_index = one_lane_index;
//						the_one_veh = first_vehicle_ID;
////						return gpu_data->lane_pool.vehicle_space[0][one_lane_index];
//					}
//				} else {
//					int next_lane_index = vpool_cpu[first_vehicle_ID].path_code[vpool_cpu[first_vehicle_ID].next_path_index];
//
//					/**
//					 * Condition 6: The Next Lane has input capacity
//					 * ---       7: The next lane has empty space
//					 */
//					if (gpu_data->lane_pool.input_capacity[next_lane_index] > 0 && gpu_data->lane_pool.predicted_empty_space[next_lane_index] > kVehicleLength) {
//						if (time_diff > maximum_waiting_time) {
//							maximum_waiting_time = time_diff;
//							*lane_index = one_lane_index;
//							the_one_veh = first_vehicle_ID;
////								return gpu_data->lane_pool.vehicle_space[0][one_lane_index];
//						}
//					} else {
//						gpu_data->lane_pool.blocked[one_lane_index] = true;
//					}
//				}
//			}
//		}
//	}
//
//	return the_one_veh;
//}
//
//void supply_simulation_vehicle_passing(int time_step) {
//	//for each node
//	for (unsigned int node_index = 0; node_index < the_network->node_size; node_index++) {
//
//		//for each capacity
//		for (int i = 0; i < gpu_data->node_pool.max_acc_flow[node_index]; i++) {
//			int lane_index = -1;
//
//			//Find A vehicle
//			int vehicle_passing_index = GetNextVehicleAtNode(node_index, &lane_index);
//
//			if (vehicle_passing_index < 0 || lane_index < 0) {
//				//			printf("one_v == NULL\n");
//				break;
//			}
//
//			if (vpool_cpu[vehicle_passing_index].entry_time <= gpu_data->lane_pool.Tp[lane_index]) {
//				gpu_data->lane_pool.queue_length[lane_index] -= kVehicleLength;
//			}
//
//			//Insert to next Lane
//			//		if (gpu_data->lane_pool.vehicle_space[0][lane_index]->next_path_index >= gpu_data->lane_pool.vehicle_space[0][lane_index]->whole_path_length) {
//			int vehicle_index = gpu_data->lane_vehicle_pool.vehicle_space[gpu_data->lane_pool.vehicle_start_index[lane_index]];
//			if (vpool_cpu[vehicle_index].next_path_index >= vpool_cpu[vehicle_index].whole_path_length) {
//				//the vehicle has finished the trip
//
//				//			printf("vehicle %d finish trip at node %d,\n", one_v->vehicle_ID, node_index);
//			} else {
//				int next_lane_index = vpool_cpu[vehicle_index].path_code[vpool_cpu[vehicle_index].next_path_index];
//				vpool_cpu[vehicle_index].next_path_index++;
//
//				//it is very critical to update the entry time when passing
//				vpool_cpu[vehicle_index].entry_time = time_step;
//
//				//add the vehicle
//				//			gpu_data->lane_pool.vehicle_passed_space[gpu_data->lane_pool.vehicle_passed_to_the_lane_counts[next_lane_index]][next_lane_index] = vehicle_passing_index;
//				int buffer_vehicle_index = gpu_data->lane_pool.buffered_vehicle_start_index[next_lane_index] + gpu_data->lane_pool.vehicle_passed_to_the_lane_counts[next_lane_index];
//				gpu_data->lane_buffered_vehicle_pool.buffered_vehicle_space[buffer_vehicle_index] = vehicle_passing_index;
//				gpu_data->lane_pool.vehicle_passed_to_the_lane_counts[next_lane_index]++;
//
//				gpu_data->lane_pool.input_capacity[next_lane_index]--;
//				gpu_data->lane_pool.predicted_empty_space[next_lane_index] -= kVehicleLength;
//
//				//printf("time_step=%d,one_v->vehicle_ID=%d,lane_index=%d, next_lane_index=%d, next_lane_index=%d\n", time_step, one_v->vehicle_ID, lane_index, next_lane_index, next_lane_index);
//			}
//
//			//Remove from current Lane
//			int start_vehicle_pool_index = gpu_data->lane_pool.vehicle_start_index[lane_index];
//			for (int j = 1; j < gpu_data->lane_pool.vehicle_counts[lane_index]; j++) {
//				gpu_data->lane_vehicle_pool.vehicle_space[start_vehicle_pool_index + j - 1] = gpu_data->lane_vehicle_pool.vehicle_space[start_vehicle_pool_index + j];
//			}
//
//			gpu_data->lane_pool.vehicle_counts[lane_index]--;
//			gpu_data->lane_pool.output_capacity[lane_index]--;
//			gpu_data->lane_pool.flow[lane_index]++;
//		}
//	}
//}
//
//void supply_simulation_after_vehicle_passing(int time_step) {
//	for (unsigned int lane_index = 0; lane_index < the_network->link_size; lane_index++) {
//
//		for (int i = 3; i > 0; i--) {
//			gpu_data->lane_pool.his_queue_length[i][lane_index] = gpu_data->lane_pool.his_queue_length[i - 1][lane_index];
//		}
//		gpu_data->lane_pool.his_queue_length[0][lane_index] = gpu_data->lane_pool.queue_length[lane_index];
//
//		//update the empty space
//		//			if (gpu_data->lane_pool.new_vehicle_join_counts[lane_index] > 0) {
//		//				gpu_data->lane_pool.empty_space[lane_index] = std::min(gpu_data->lane_pool.speed[lane_index] * UNIT_TIME_STEPS, gpu_data->lane_pool.empty_space[lane_index])
//		//				if (gpu_data->lane_pool.empty_space[lane_index] < 0) gpu_data->lane_pool.empty_space[lane_index] = 0;
//		//			}
//		//			else {
//		gpu_data->lane_pool.empty_space[lane_index] = gpu_data->lane_pool.empty_space[lane_index] + gpu_data->lane_pool.speed[lane_index] * kUnitTimeStep;
//		gpu_data->lane_pool.empty_space[lane_index] = std::min(gpu_data->lane_pool.lane_length[lane_index] - gpu_data->lane_pool.queue_length[lane_index], gpu_data->lane_pool.empty_space[lane_index]);
//
//	}
//}
