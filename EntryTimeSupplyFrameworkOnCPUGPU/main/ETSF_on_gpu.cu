/**
 *This project targets to check GPU is an option for DynaMIT.
 *This project also targets for a paper "Mesoscopic Traffic Simulation on GPU"
 */

#include "../components_on_cpu/network/network.h"
#include "../components_on_cpu/demand/od_pair.h"
#include "../components_on_cpu/demand/od_path.h"
#include "../components_on_cpu/demand/vehicle.h"
#include "../components_on_cpu/util/time_tools.h"
#include "../components_on_cpu/util/string_tools.h"
#include "../components_on_cpu/util/simulation_results.h"
#include "../components_on_cpu/util/shared_cpu_include.h"

#include "../components_on_gpu/supply/on_GPU_memory.h"
#include "../components_on_gpu/supply/on_GPU_vehicle.h"
#include "../components_on_gpu/supply/on_GPU_new_lane_vehicles.h"
#include "../components_on_gpu/util/shared_gpu_include.h"
#include "../components_on_gpu/util/on_gpu_configuration.h"
#include "../components_on_gpu/on_GPU_kernal.cuh"
#include "../components_on_gpu/on_GPU_Macro.h"

using namespace std;

/**
 * CUDA Execution Configuration
 */
int road_blocks;
const int road_threads_in_a_block = 192;
int node_blocks;
const int node_threads_in_a_block = 192;

/*
 * Demand
 */
Network* the_network;
std::vector<ODPair*> all_od_pairs;
std::vector<ODPairPATH*> all_od_paths;
std::vector<Vehicle*> all_vehicles;

/*
 * Path Input Config
 */
//std::string network_file_path = "data_inputs/exp1_network/network_10.dat";
//std::string demand_file_path = "data_inputs/exp1/demand_10_50000.dat";
//std::string od_pair_file_path = "data_inputs/exp1/od_pair_10.dat";
//std::string od_pair_paths_file_path = "data_inputs/exp1/od_pair_paths_10.dat";
std::string network_file_path = "data_inputs/exp2/network_100.dat";
std::string demand_file_path = "data_inputs/exp2/demand_100_100000.dat";
std::string od_pair_file_path = "data_inputs/exp2/od_pair_100.dat";
std::string od_pair_paths_file_path = "data_inputs/exp2/od_pair_cleaned_paths_100.dat";

/*
 * All data in GPU
 */
GPUMemory* gpu_data;
GPUSharedParameter* parameter_seeting_on_gpu;
#ifdef ENABLE_CONSTANT_MEMORY
__constant__ GPUSharedParameter data_setting_gpu_constant;
#endif

//A large memory space is pre-defined in order to copy to GPU
GPUVehicle *vpool_cpu;
GPUVehicle *vpool_gpu;

int *vpool_cpu_index;
int *vpool_gpu_index;

/**
 * Simulation Results
 */
std::string simulation_output_file_path = "output/test3.txt";
std::map<int, SimulationResults*> simulation_results_pool;
ofstream simulation_results_output_file;

//buffer is only used when kGPUToCPUSimulationResultsCopyBufferSize > 1
SimulationResults* simulation_results_buffer_on_gpu;

//Used for buffer at CPU side
SimulationResults* one_buffer = NULL;

/*
 * GPU Streams
 * stream1: GPU Supply Simulation
 */
cudaStream_t stream_gpu_supply;
cudaStream_t stream_gpu_io;
cudaEvent_t gpu_supply_one_tick_simulation_done_trigger_event;

/*
 * Time Management
 */
long simulation_start_time;
long simulation_end_time;
long simulation_time_step;

/*
 * simulation_time is already finished time;
 * simulation_time + 1 might be the current simulating time on GPU
 */
long to_simulate_time;

/*
 * simulation_results_outputed_time is already outputted time;
 * simulation_results_outputed_time + 1 might be the outputing time on CPU
 */
long to_output_simulation_result_time;

/*
 */
std::map<int, int> link_ID_to_link_Index;
std::map<int, int> link_Index_to_link_ID;
std::map<int, int> node_ID_to_node_Index;
std::map<int, int> node_index_to_node_ID;

/*
 * Define Major Functions
 */
bool InitParams(int argc, char* argv[]);
bool LoadInNetwork();
bool LoadInDemand();
bool InitilizeCPU();
bool InitilizeGPU();
bool InitGPUParameterSetting(GPUSharedParameter* data_setting_gpu);
bool InitGPUData(GPUMemory* data_local);
bool StartSimulation();
bool DestoryResources();

/*
 * Define Helper Functions
 */
StringTools* str_tools;
bool CopySimulatedResultsToCPU(int time_step);
bool CopyBufferSimulatedResultsToCPU(int time_step);
bool OutputSimulatedResults(int time_step);
bool OutputBufferedSimulatedResults(int time_step);

inline int TimestepToArrayIndex(int time_step) {
	return (time_step - kStartTimeSteps) / kUnitTimeStep;
}

/*
 * MAIN
 */
int main(int argc, char* argv[]) {

	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	if (InitParams(argc, argv) == false) {
		cout << "InitParams fails" << endl;
		return 0;
	}

	if (LoadInNetwork() == false) {
		cout << "Loading network fails" << endl;
		return 0;
	}

	if (LoadInDemand() == false) {
		cout << "Loading demand fails" << endl;
		return 0;
	}

	if (InitilizeCPU() == false) {
		cout << "InitilizeCPU fails" << endl;
		return 0;
	}

	if (InitilizeGPU() == false) {
		cout << "InitilizeGPU fails" << endl;
		return 0;
	}

	//create streams
	cudaStreamCreate(&stream_gpu_supply);
	cudaStreamCreate(&stream_gpu_io);

	//create a event
	cudaEventCreate(&gpu_supply_one_tick_simulation_done_trigger_event);

	std::cout << "Simulation Starts" << std::endl;

	TimeTools profile;
	profile.start_profiling();

	//Start Simulation (ETSF implemented inside)
	if (StartSimulation() == false) {
		cout << "Simulation Fails" << endl;
		DestoryResources();
		return 0;
	}

	profile.end_profiling();
	profile.output();

	DestoryResources();
	cout << "Simulation Succeed!" << endl;

#ifdef _WIN32
	system("pause");
#endif

	return 0;
}

/**
 *
 */
bool InitParams(int argc, char* argv[]) {
	if (argc == 5) {
		network_file_path = argv[2];
		demand_file_path = argv[3];
		simulation_output_file_path = argv[4];
		std::cout << "parameters updated" << std::endl;
	}
	return true;
}
bool LoadInNetwork() {
	the_network = new Network();

	the_network->all_links.clear();
	the_network->all_nodes.clear();
	the_network->node_mapping.clear();

	return Network::load_network(*the_network, network_file_path);
}

bool LoadInDemand() {
	if (ODPair::load_in_all_ODs(all_od_pairs, od_pair_file_path) == false) {
		return false;
	}
	if (ODPairPATH::load_in_all_OD_Paths(all_od_paths, od_pair_paths_file_path) == false) {
		return false;
	}
	if (Vehicle::load_in_all_vehicles(all_vehicles, demand_file_path) == false) {
		return false;
	}
	return true;
}

bool InitilizeCPU() {
	simulation_start_time = kStartTimeSteps;
	simulation_end_time = kEndTimeSteps; // 2 hours
	simulation_time_step = kUnitTimeStep;

	assert(simulation_time_step == 1);

	to_simulate_time = simulation_start_time;
	to_output_simulation_result_time = simulation_start_time;

	road_blocks = kLaneSize / road_threads_in_a_block + 1;
	node_blocks = kNodeSize / node_threads_in_a_block + 1;

	simulation_results_pool.clear();
	simulation_results_output_file.open(simulation_output_file_path.c_str());
	simulation_results_output_file << "##TIME STEP" << ":Lane ID:" << ":(" << "COUNTS" << ":" << "flow" << ":" << "density" << ":" << "speed" << ":" << "queue_length" << ")" << endl;
	str_tools = new StringTools();

	return true;
}

bool InitilizeGPU() {
	gpu_data = NULL;
	parameter_seeting_on_gpu = NULL;

	GPUMemory* data_local = new GPUMemory();
	InitGPUData(data_local);

	GPUSharedParameter* data_setting_gpu = new GPUSharedParameter();
	InitGPUParameterSetting(data_setting_gpu);

#ifdef ENABLE_CONSTANT_MEMORY
	GPUSharedParameter data_setting_cpu_constant;
	InitGPUParameterSetting(&data_setting_cpu_constant);
#endif

	//apply memory on GPU
	size_t memory_space_for_vehicles = all_vehicles.size() * sizeof(GPUVehicle);
	if (cudaMalloc((void**) &vpool_gpu, memory_space_for_vehicles) != cudaSuccess) {
		cerr << "cudaMalloc((void**) &vpool_gpu, memory_space_for_vehicles) failed" << endl;
	}

	size_t memory_space_for_rebuild_index = kTotalTimeSteps * kLaneSize * kVehicleMaxLoadingOneTime * sizeof(int);
	if (cudaMalloc((void**) &vpool_gpu_index, memory_space_for_rebuild_index) != cudaSuccess) {
		cerr << "cudaMalloc((void**) &vpool_gpu_index, memory_space_for_rebuild_index) failed" << endl;
	}

	if (cudaMalloc((void**) &gpu_data, data_local->total_size()) != cudaSuccess) {
		cerr << "cudaMalloc(&gpu_data, sizeof(GPUMemory)) failed" << endl;
	}

	if (cudaMalloc((void**) &parameter_seeting_on_gpu, sizeof(GPUSharedParameter)) != cudaSuccess) {
		cerr << "cudaMalloc(&GPUSharedParameter, sizeof(GPUSharedParameter)) failed" << endl;
	}

#ifdef ENABLE_CONSTANT_MEMORY
	cudaMemcpyToSymbol(data_setting_gpu_constant, &data_setting_cpu_constant, sizeof(GPUSharedParameter));
#endif

	//apply a buffer space for GPU outputs
	if (kGPUToCPUSimulationResultsCopyBufferSize > 1) {
		size_t memory_space_for_buffer_outputs = sizeof(SimulationResults) * kGPUToCPUSimulationResultsCopyBufferSize;
		if (cudaMalloc((void**) &simulation_results_buffer_on_gpu, memory_space_for_buffer_outputs) != cudaSuccess) {
			cerr << "cudaMalloc((void**) &simulation_results_buffer_on_gpu, memory_space_for_buffer_outputs) failed" << endl;
		}
	}

	cudaMemcpy(vpool_gpu, vpool_cpu, memory_space_for_vehicles, cudaMemcpyHostToDevice);
	cudaMemcpy(vpool_gpu_index, vpool_cpu_index, memory_space_for_rebuild_index, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_data, data_local, data_local->total_size(), cudaMemcpyHostToDevice);
	cudaMemcpy(parameter_seeting_on_gpu, data_setting_gpu, sizeof(GPUSharedParameter), cudaMemcpyHostToDevice);

	int GRID_SIZE = 1;
	int BLOCK_SIZE = kTotalTimeSteps;

	LinkGPUData<<<GRID_SIZE, BLOCK_SIZE>>>(gpu_data, kTotalTimeSteps, vpool_gpu, vpool_gpu_index, parameter_seeting_on_gpu);

	//wait for all CUDA related operations to finish;
	std::cout << "LinkGPUData begins" << std::endl;
	cudaDeviceSynchronize();
	std::cout << "LinkGPUData ends" << std::endl;

#ifdef ENABLE_OUTPUT_GPU_BUFFER
	cudaMallocHost((void **) &one_buffer, sizeof(SimulationResults) * kGPUToCPUSimulationResultsCopyBufferSize);
#endif

	return true;
}

/*
 * Copy the parameter setting to GPU memory
 */
bool InitGPUParameterSetting(GPUSharedParameter* data_setting_gpu) {
	data_setting_gpu->kOnGPULaneSize = kLaneSize;
	data_setting_gpu->kOnGPUNodeSize = kNodeSize;

	data_setting_gpu->kOnGPUEndTimeStep = kEndTimeSteps;
	data_setting_gpu->kOnGPUStartTimeStep = kStartTimeSteps;
	data_setting_gpu->kOnGPUTotalTimeSteps = kTotalTimeSteps;
	data_setting_gpu->kOnGPUUnitTimeStep = kUnitTimeStep;
	data_setting_gpu->kOnGPUVehicleLength = kVehicleLength;

	data_setting_gpu->kOnGPURoadLength = kRoadLength;
	data_setting_gpu->kOnGPULaneInputCapacityPerTimeStep = kLaneInputCapacityPerTimeStep;
	data_setting_gpu->kOnGPULaneOutputCapacityPerTimeStep = kLaneOutputCapacityPerTimeStep;
	data_setting_gpu->kOnGPUMaxLaneCodingLength = kMaxLaneCodingLength;
	data_setting_gpu->kOnGPUMaxlaneDownstream = kMaxLaneDownstream;
	data_setting_gpu->kOnGPUMaxLaneUpstream = kMaxLaneUpstream;
	data_setting_gpu->kOnGPUMaxRouteLength = kMaxRouteLength;
	data_setting_gpu->kOnGPUMaxVehiclePerLane = kMaxVehiclePerLane;
	data_setting_gpu->kOnGPUVehicleMaxLoadingOneTime = kVehicleMaxLoadingOneTime;

	data_setting_gpu->kOnGPUAlpha = kAlpha;
	data_setting_gpu->kOnGPUBeta = kBeta;
	data_setting_gpu->kOnGPUMaxSpeed = kMaxSpeed;
	data_setting_gpu->kOnGPUMinSpeed = kMinSpeed;
	data_setting_gpu->kOnGPUMaxDensity = kMaxDensity;
	data_setting_gpu->kOnGPUMinDensity = kMinDensity;

	data_setting_gpu->kOnGPUGPUToCPUSimulationResultsCopyBufferSize = kGPUToCPUSimulationResultsCopyBufferSize;

	return true;
}

/*
 * Build a GPU data
 */
bool InitGPUData(GPUMemory* data_local) {

	node_ID_to_node_Index.clear();
	link_ID_to_link_Index.clear();
	node_index_to_node_ID.clear();
	link_Index_to_link_ID.clear();

	/**
	 * First Part: Lane
	 */

	for (int i = 0; i < the_network->all_links.size(); i++) {
		Link* one_link = the_network->all_links[i];

		data_local->lane_pool.lane_ID[i] = one_link->link_id;

		link_ID_to_link_Index[one_link->link_id] = i;
		link_Index_to_link_ID[i] = one_link->link_id;
		//make sure assert is working
//		assert(1 == 0);

//		assert(one_link->link_id == i);

		data_local->lane_pool.from_node_id[i] = one_link->from_node->node_id;
		data_local->lane_pool.to_node_id[i] = one_link->to_node->node_id;

		data_local->lane_pool.Tp[i] = simulation_start_time - simulation_time_step;
		data_local->lane_pool.Tq[i] = simulation_start_time - simulation_time_step;
		data_local->lane_pool.accumulated_offset[i] = 0;

		data_local->lane_pool.flow[i] = 0;
		data_local->lane_pool.density[i] = 0;
		data_local->lane_pool.speed[i] = 0;
		data_local->lane_pool.queue_length[i] = 0;

		/*
		 * for density calculation
		 */
		data_local->lane_pool.lane_length[i] = kRoadLength; // meter
		data_local->lane_pool.max_vehicles[i] = kRoadLength / kVehicleLength; //number of vehicles
		data_local->lane_pool.output_capacity[i] = kLaneOutputCapacityPerTimeStep; //
		data_local->lane_pool.input_capacity[i] = kLaneInputCapacityPerTimeStep; //
		data_local->lane_pool.empty_space[i] = kRoadLength;

		/*
		 * for speed calculation
		 */
		data_local->lane_pool.alpha[i] = kAlpha;
		data_local->lane_pool.beta[i] = kBeta;
		data_local->lane_pool.max_density[i] = kMaxDensity;
		data_local->lane_pool.min_density[i] = kMinDensity;
		data_local->lane_pool.MAX_speed[i] = kMaxSpeed;
		data_local->lane_pool.MIN_speed[i] = kMinSpeed;

		data_local->lane_pool.vehicle_counts[i] = 0;
		data_local->lane_pool.vehicle_passed_to_the_lane_counts[i] = 0;

		for (int c = 0; c < kMaxVehiclePerLane; c++) {
			data_local->lane_pool.vehicle_space[c][i] = NULL;
		}

		for (int c = 0; c < kLaneInputCapacityPerTimeStep; c++) {
			data_local->lane_pool.vehicle_passed_space[c][i] = NULL;
		}

		for (int j = 0; j < kTotalTimeSteps; j++) {
			data_local->lane_pool.speed_history[j][i] = -1;
		}

		//it is assumed that QUEUE_LENGTH_HISTORY = 4;
		assert(kQueueLengthHistory == 4);
		float weight[kQueueLengthHistory];
		weight[0] = 1.0;
		weight[1] = 0;
		weight[2] = 0;
		weight[3] = 0;

		//		{ 0.2, 0.3, 0.5, 0 };

		for (int j = 0; j < kQueueLengthHistory; j++) {
			data_local->lane_pool.his_queue_length[j][i] = -1;
			data_local->lane_pool.his_queue_length_weighting[j][i] = weight[j];
		}

		data_local->lane_pool.predicted_empty_space[i] = 0;
		data_local->lane_pool.predicted_queue_length[i] = 0;
		data_local->lane_pool.last_time_empty_space[i] = 0;
	}

	/**
	 * Second Part: Node
	 */
//	NodePool* the_node_pool = data_local->node_pool;
	for (int i = 0; i < the_network->all_nodes.size(); i++) {
		Node* one_node = the_network->all_nodes[i];
		node_ID_to_node_Index[one_node->node_id] = i;
		node_index_to_node_ID[i] = one_node->node_id;

		data_local->node_pool.node_ID[i] = one_node->node_id;
		data_local->node_pool.max_acc_flow[i] = 0;
		data_local->node_pool.acc_upstream_capacity[i] = 0;
		data_local->node_pool.acc_downstream_capacity[i] = 0;

		for (int j = 0; j < kMaxLaneUpstream; j++) {
			data_local->node_pool.upstream[j][i] = -1;
		}

		for (int j = 0; j < one_node->upstream_links.size(); j++) {
			int link_index = link_ID_to_link_Index[one_node->upstream_links[j]->link_id];
			data_local->node_pool.upstream[j][i] = link_index;
			data_local->node_pool.acc_upstream_capacity[i] += kLaneInputCapacityPerTimeStep;
		}

		for (int j = 0; j < kMaxLaneDownstream; j++) {
			data_local->node_pool.downstream[j][i] = -1;
		}

		for (int j = 0; j < one_node->downstream_links.size(); j++) {
			int link_index = link_ID_to_link_Index[one_node->downstream_links[j]->link_id];
			data_local->node_pool.downstream[j][i] = link_index;
			data_local->node_pool.acc_downstream_capacity[i] += kLaneOutputCapacityPerTimeStep;
		}

		if (data_local->node_pool.acc_upstream_capacity[i] <= 0) {
			data_local->node_pool.max_acc_flow[i] = data_local->node_pool.acc_downstream_capacity[i];
		} else if (data_local->node_pool.acc_downstream_capacity[i] <= 0) {
			data_local->node_pool.max_acc_flow[i] = data_local->node_pool.acc_upstream_capacity[i];
		} else {
			data_local->node_pool.max_acc_flow[i] = std::min(data_local->node_pool.acc_upstream_capacity[i], data_local->node_pool.acc_downstream_capacity[i]);
		}
	}

	/**
	 * Third Part:
	 */

//Init VehiclePool
	for (int i = kStartTimeSteps; i < kEndTimeSteps; i += kUnitTimeStep) {
		for (int j = 0; j < kLaneSize; j++) {
			data_local->new_vehicles_every_time_step[i].new_vehicle_size[j] = 0;
			data_local->new_vehicles_every_time_step[i].lane_ID[j] = -1;
		}
	}

	std::cout << "all_vehicles.size():" << all_vehicles.size() << std::endl;

//init host vehicle pool data /*xiaosong*/
	int memory_space_for_vehicles = all_vehicles.size() * sizeof(GPUVehicle);
	vpool_cpu = (GPUVehicle*) malloc(memory_space_for_vehicles);
	if (vpool_cpu == NULL)
		exit(1);

	std::cout << "vpool_cpu_index size:" << kTotalTimeSteps * kLaneSize * kVehicleMaxLoadingOneTime << std::endl;
	vpool_cpu_index = (int*) malloc(kTotalTimeSteps * kLaneSize * kVehicleMaxLoadingOneTime * sizeof(int));
	if (vpool_cpu_index == NULL)
		exit(1);

	for (int i = kStartTimeSteps; i < kEndTimeSteps; i += kUnitTimeStep) {
		for (int j = 0; j < kLaneSize; j++) {
			for (int z = 0; z < kVehicleMaxLoadingOneTime; z++) {
				int index_t = i * kLaneSize * kVehicleMaxLoadingOneTime + j * kVehicleMaxLoadingOneTime + z;

				//-1 means there is no vehicle on the road
				vpool_cpu_index[index_t] = -1;
			}
		}
	}

	int nVehiclePerTick = kVehicleMaxLoadingOneTime * kLaneSize;
	std::cout << "init all_vehicles" << std::endl;

	int total_inserted_vehicles = 0;

//Insert Vehicles
	for (int i = 0; i < all_vehicles.size(); i++) {
		Vehicle* one_vehicle = all_vehicles[i];

		int time_index = one_vehicle->entry_time;
		int time_index_covert = TimestepToArrayIndex(time_index);
		assert(time_index == time_index_covert);
		//try to load vehicles beyond the simulation border
		if (time_index_covert >= kTotalTimeSteps)
			continue;

		int lane_ID = all_od_paths[one_vehicle->path_id]->link_ids[0];
		int lane_Index = link_ID_to_link_Index[lane_ID];

		if (data_local->new_vehicles_every_time_step[time_index_covert].new_vehicle_size[lane_Index] < kVehicleMaxLoadingOneTime) {
			int last_vehicle_index = data_local->new_vehicles_every_time_step[time_index_covert].new_vehicle_size[lane_Index];
			int idx_vpool = time_index_covert * nVehiclePerTick + lane_Index * kVehicleMaxLoadingOneTime + last_vehicle_index;

			//for gpu to rebuild
			vpool_cpu_index[idx_vpool] = i;
//			std::cout << "idx_vpool:" << idx_vpool << " is map to i:" << i << std::endl;

			vpool_cpu[i].vehicle_ID = one_vehicle->vehicle_id;
			vpool_cpu[i].entry_time = time_index;
			vpool_cpu[i].current_lane_ID = lane_Index;
			int max_copy_length = kMaxRouteLength > all_od_paths[one_vehicle->path_id]->link_ids.size() ? all_od_paths[one_vehicle->path_id]->link_ids.size() : kMaxRouteLength;

			for (int p = 0; p < max_copy_length; p++) {
//				vpool_cpu[i].path_code[p] = all_od_paths[one_vehicle->path_id]->route_code[p] ? 1 : 0;
				vpool_cpu[i].path_code[p] = link_ID_to_link_Index[all_od_paths[one_vehicle->path_id]->link_ids[p]];
			}

			//ready for the next lane, so next_path_index is set to 1, if the next_path_index == whole_path_length, it means cannot find path any more, can exit;
			vpool_cpu[i].next_path_index = 1;
			vpool_cpu[i].whole_path_length = all_od_paths[one_vehicle->path_id]->link_ids.size();

			//will be re-writen by GPU
			data_local->new_vehicles_every_time_step[time_index_covert].new_vehicles[lane_Index][last_vehicle_index] = &(vpool_cpu[i]);
			data_local->new_vehicles_every_time_step[time_index_covert].new_vehicle_size[lane_Index]++;

			total_inserted_vehicles++;
		} else {
//			std::cout << "Loading Vehicles Exceeds The Loading Capacity: Time:" << time_index_covert << ", Lane_ID:" << lane_ID << ",i:" << i << ",ID:" << one_vehicle->vehicle_id << std::endl;
		}
	}

	std::cout << "init all_vehicles:" << total_inserted_vehicles << std::endl;

	return true;
}

bool DestoryResources() {
	simulation_results_output_file.flush();
	simulation_results_output_file.close();

	if (vpool_cpu != NULL)
		delete vpool_cpu;
	if (str_tools != NULL)
		delete str_tools;

	cudaDeviceReset();
	return true;
}

bool StartSimulation() {
	bool first_time_step = true;

	/*
	 * Simulation Loop
	 */

	while (((to_simulate_time >= simulation_end_time) && (to_output_simulation_result_time >= simulation_end_time)) == false) {

		//GPU has done simulation at current time
		if (to_simulate_time < simulation_end_time && (cudaEventQuery(gpu_supply_one_tick_simulation_done_trigger_event) == cudaSuccess)) {
			if (first_time_step == true) {
				first_time_step = false;
			} else {
#ifdef ENABLE_OUTPUT_GPU_BUFFER
				if ((to_simulate_time + 1) % kGPUToCPUSimulationResultsCopyBufferSize == 0) {
					CopyBufferSimulatedResultsToCPU(to_simulate_time);
				}
#else
				CopySimulatedResultsToCPU(to_simulate_time);
#endif

				to_simulate_time += simulation_time_step;
			}

			if (to_simulate_time < simulation_end_time) {
#ifdef ENABLE_OUTPUT
				cout << "to_simulate_time:" << to_simulate_time << ", simulation_end_time:" << simulation_end_time << endl;
#endif

				SupplySimulationPreVehiclePassing<<<road_blocks, road_threads_in_a_block, 0, stream_gpu_supply>>>(gpu_data, to_simulate_time, kLaneSize, parameter_seeting_on_gpu);
				SupplySimulationVehiclePassing<<<node_blocks, node_threads_in_a_block, 0, stream_gpu_supply>>>(gpu_data, to_simulate_time, kNodeSize, parameter_seeting_on_gpu);
				SupplySimulationAfterVehiclePassing<<<road_blocks, road_threads_in_a_block, 0, stream_gpu_supply>>>(gpu_data, to_simulate_time, kLaneSize, parameter_seeting_on_gpu);

#ifdef ENABLE_OUTPUT_GPU_BUFFER
				supply_simulated_results_to_buffer<<<road_blocks, road_threads_in_a_block, 0, stream_gpu_supply>>>(gpu_data, to_simulate_time, kLaneSize, simulation_results_buffer_on_gpu,
						parameter_seeting_on_gpu);
#endif
				cudaEventRecord(gpu_supply_one_tick_simulation_done_trigger_event, stream_gpu_supply);
			}
		}
		//GPU is busy, so CPU does something else (I/O)
		else if (to_output_simulation_result_time < to_simulate_time) {

#ifdef ENABLE_OUTPUT_GPU_BUFFER
			if (to_output_simulation_result_time <= to_simulate_time - kGPUToCPUSimulationResultsCopyBufferSize) {

#ifdef ENABLE_OUTPUT
				OutputBufferedSimulatedResults(to_output_simulation_result_time);
#endif
				to_output_simulation_result_time += simulation_time_step * kGPUToCPUSimulationResultsCopyBufferSize;
			}
#else
#ifdef ENABLE_OUTPUT
			OutputSimulatedResults(to_output_simulation_result_time);
#endif
			to_output_simulation_result_time += simulation_time_step;
#endif
		} else {

#ifdef ENABLE_OUTPUT
			cout << "---------------------" << endl;
			cout << "CPU nothing to do" << endl;
			cout << "to_simulate_time:" << to_simulate_time << endl;
			cout << "to_output_simulation_result_time:" << to_output_simulation_result_time << endl;
			cout << "---------------------" << endl;
#endif
		}
	}

	return true;
}

/**
 * Minor Functions
 */
bool CopySimulatedResultsToCPU(int time_step) {
	int index = TimestepToArrayIndex(time_step);
	SimulationResults* one = new SimulationResults();

	cudaMemcpy(one->flow, gpu_data->lane_pool.flow, sizeof(float) * kLaneSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(one->density, gpu_data->lane_pool.density, sizeof(float) * kLaneSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(one->speed, gpu_data->lane_pool.speed, sizeof(float) * kLaneSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(one->queue_length, gpu_data->lane_pool.queue_length, sizeof(float) * kLaneSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(one->counts, gpu_data->lane_pool.vehicle_counts, sizeof(int) * kLaneSize, cudaMemcpyDeviceToHost);
	simulation_results_pool[index] = one;

	return true;
}

bool CopyBufferSimulatedResultsToCPU(int time_step) {
	cudaMemcpyAsync(one_buffer, simulation_results_buffer_on_gpu, sizeof(SimulationResults) * kGPUToCPUSimulationResultsCopyBufferSize, cudaMemcpyDeviceToHost, stream_gpu_io);

	for (int i = 0; i < kGPUToCPUSimulationResultsCopyBufferSize; i++) {
		int time_index = time_step - (kGPUToCPUSimulationResultsCopyBufferSize - 1) + i;
		simulation_results_pool[time_index] = &one_buffer[i];
	}

	return true;
}

bool OutputSimulatedResults(int time_step) {
	if (simulation_results_pool.find(time_step) == simulation_results_pool.end()) {
		std::cerr << "System Error, Try to output time " << time_step << ", while it is not ready!" << std::endl;
		return false;
	}

	int index = TimestepToArrayIndex(time_step);
	SimulationResults* one = simulation_results_pool[index];
	assert(one != NULL);

	for (int i = 0; i < kLaneSize; i++) {
		int lane_ID = i;
		int lane_Index = link_ID_to_link_Index[lane_ID];

		simulation_results_output_file << time_step << ":lane:" << lane_ID << ":(" << one->counts[lane_Index] << ":" << one->flow[lane_Index] << ":" << one->density[lane_Index]
//				<< ":" << gpu_data->lane_pool.speed[i] << ":" << gpu_data->lane_pool.queue_length[i] << ":" << gpu_data->lane_pool.empty_space[i] << ")" << endl;
				<< ":" << one->speed[lane_Index] << ":" << one->queue_length[lane_Index] << ")" << endl;
	}

	return true;
}

bool OutputBufferedSimulatedResults(int time_step) {
	std::cout << "OutputBufferedSimulatedResults AT time " << time_step << std::endl;

	for (int i = 0; i < kGPUToCPUSimulationResultsCopyBufferSize; i++) {
		OutputSimulatedResults(time_step + i);
	}

	return true;
}
