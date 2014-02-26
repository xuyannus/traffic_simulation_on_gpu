/**
 *This project targets to check GPU is an option for DynaMIT.
 *This project also targets for a paper "Mesoscopic Traffic Simulation on GPU"
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "../on_cpu/network/Network.h"
#include "../on_cpu/demand/OD_Pair.h"
#include "../on_cpu/demand/OD_Path.h"
#include "../on_cpu/demand/Vehicle.h"
#include "../on_cpu/util/TimeTools.h"
#include "../on_cpu/util/StringTools.h"

#include "../on_cpu/util/shared_cpu_include.h"
#include "../on_gpu/supply/kernel_functions.h"
#include "../on_gpu/supply/OnGPUMemory.h"
#include "../on_cpu/util/SimulationResults.h"
#include "../on_gpu/supply/OnGPUVehicle.h"
#include "../on_gpu/supply/OnGPUNewLaneVehicles.h"
#include "../on_gpu/util/OnGPUData.h"

#define ENABLE_OUTPUT
#define ENABLE_CONSTANT
#define ENABLE_MORE_REGISTER
//#define ENABLE_OUTPUT_GPU_BUFFER

using namespace std;

/**
 * CUDA Execution Configuration
 */
int roadBlocks;
int roadThreadsInABlock;

int nodeBlocks;
int nodeThreadsInABlock;

/*
 * Demand
 */
Network* the_network;
vector<OD_Pair*> all_od_pairs;
vector<OD_Pair_PATH*> all_od_paths;
vector<Vehicle*> all_vehicles;

/*
 * Path Input Config
 */
//std::string network_file_path = "data/exp1_network/network_10_rank.dat";
//std::string demand_file_path = "data/exp1/demand_10_10000.dat";
//std::string od_pair_file_path = "data/exp1/od_pair_10.dat";
//std::string od_pair_paths_file_path = "data/exp1/od_pair_paths_10.dat";

//std::string network_file_path = "data/exp2/network_100_rank.dat";
//std::string network_file_path = "data/exp2/network_100_congestion_rank.dat";

std::string network_file_path = "data/exp2/network_100.dat";
std::string demand_file_path = "data/exp2/demand_100_100000.dat";
std::string od_pair_file_path = "data/exp2/od_pair_100.dat";
std::string od_pair_paths_file_path = "data/exp2/od_pair_cleaned_paths_100.dat";

//std::string network_file_path = "data/exp2_rank/network_100_GPU_ran_reform.dat";
//std::string demand_file_path = "data/exp2_rank/demand_100_100000.dat";
//std::string od_pair_file_path = "data/exp2_rank/od_pair_100_ran.dat";
//std::string od_pair_paths_file_path = "data/exp2_rank/od_pair_cleaned_paths_100_ran.dat";

//std::string network_file_path = "data/exp2_rank/network_100_GPU_flow_reform.dat";
//std::string demand_file_path = "data/exp2_rank/demand_100_100000.dat";
//std::string od_pair_file_path = "data/exp2_rank/od_pair_100.dat";
//std::string od_pair_paths_file_path = "data/exp2_rank/od_pair_cleaned_paths_100_updated.dat";

/*
 * All data in GPU
 */
GPUMemory* gpu_data;
GPUSharedParameter* parameter_seeting_on_gpu;
#ifdef ENABLE_CONSTANT
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
std::string simulation_output_file_path = "output/simulated_outputs_gpu_100.txt.compare";
std::map<int, SimulationResults*> simulation_results_pool;
ofstream simulation_results_output_file;

//buffer is only used when GPU_TO_CPU_SIMULATION_RESULTS_COPY_BUFFER_SIZE > 1
SimulationResults* simulation_results_buffer_on_gpu;

//Used for buffer at CPU side
SimulationResults* one_buffer = NULL;

/*
 * GPU Streams
 * stream1: GPU Supply Simulation
 */
cudaStream_t stream_gpu_supply;
cudaEvent_t GPU_supply_one_time_simulation_done_event;

cudaStream_t stream_gpu_io;

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
 * Define Major Functions
 */
bool init_params(int argc, char* argv[]);
bool load_in_network();
bool load_in_demand();
bool initilizeCPU();
bool initilizeGPU();
bool initGPUParameterSetting(GPUSharedParameter* data_setting_gpu);
bool initGPUData(GPUMemory* data_local);

bool start_simulation();
bool destory_resources();

/*
 * Define Helper Functions
 */
bool copy_simulated_results_to_CPU(int time_step);
bool copy_buffer_simulated_results_to_CPU(int time_step);
bool output_simulated_results(int time_step);
bool output_buffered_simulated_results(int time_step);

StringTools* str_tools;
inline int timestep_to_arrayindex(int time_step) {
	return (time_step - START_TIME_STEPS) / UNIT_TIME_STEPS;
}

/*
 * Supply Function Define
 */

__global__ void supply_simulation_pre_vehicle_passing(GPUMemory* gpu_data, int time_step, int segment_length, GPUSharedParameter* data_setting_gpu);
__global__ void supply_simulation_vehicle_passing(GPUMemory* gpu_data, int time_step, int node_length, GPUSharedParameter* data_setting_gpu);
__global__ void supply_simulation_after_vehicle_passing(GPUMemory* gpu_data, int time_step, int segment_length, GPUSharedParameter* data_setting_gpu);
__global__ void supply_simulated_results_to_buffer(GPUMemory* gpu_data, int time_step, int segment_length, SimulationResults* buffer, GPUSharedParameter* data_setting_gpu);
__device__ GPUVehicle* get_next_vehicle_at_node(GPUMemory* gpu_data, int node_id, int* lane_id, GPUSharedParameter* data_setting_gpu);

/*
 * Utility Function
 */

__global__ void linkGPUData(GPUMemory *gpu_data, GPUVehicle *vpool_gpu, int *vpool_gpu_index, GPUSharedParameter* data_setting_gpu);

__device__ float min_device(float one_value, float the_other);
__device__ float max_device(float one_value, float the_other);

/*
 * MAIN
 */
int main(int argc, char* argv[]) {

	cudaDeviceSetCacheConfig (cudaFuncCachePreferL1);

	if (init_params(argc, argv) == false) {
		cout << "init_params fails" << endl;
		return 0;
	}

	if (load_in_network() == false) {
		cout << "Loading network fails" << endl;
		return 0;
	}

	if (load_in_demand() == false) {
		cout << "Loading demand fails" << endl;
		return 0;
	}

	if (initilizeCPU() == false) {
		cout << "InitilizeCPU fails" << endl;
		return 0;
	}

	if (initilizeGPU() == false) {
		cout << "InitilizeGPU fails" << endl;
		return 0;
	}

	//create streams
	cudaStreamCreate(&stream_gpu_supply);
	cudaStreamCreate(&stream_gpu_io);

	//create a event
	cudaEventCreate(&GPU_supply_one_time_simulation_done_event);

	std::cout << "Simulation Starts" << std::endl;

	TimeTools profile;
	profile.start_profiling();

	//Start Simulation
	if (start_simulation() == false) {
		cout << "Simulation Fails" << endl;
		destory_resources();
		return 0;
	}

	profile.end_profiling();
	profile.output();

	destory_resources();

	cout << "Simulation Succeed!" << endl;

#ifdef _WIN32
	system("pause");
#endif

	return 0;
}

/**
 *
 */
bool init_params(int argc, char* argv[]) {
	if (argc == 5) {
		network_file_path = argv[2];
		demand_file_path = argv[3];
		simulation_output_file_path = argv[4];
		std::cout << "parameters updated" << std::endl;
	}
	return true;
}
bool load_in_network() {
	the_network = new Network();

	the_network->all_links.clear();
	the_network->all_nodes.clear();
	the_network->node_mapping.clear();

	return Network::load_network(the_network, network_file_path);
}

bool load_in_demand() {

	if (OD_Pair::load_in_all_ODs(all_od_pairs, od_pair_file_path) == false) {
		return false;
	}

	if (OD_Pair_PATH::load_in_all_OD_Paths(all_od_paths, od_pair_paths_file_path) == false) {
		return false;
	}

	if (Vehicle::load_in_all_vehicles(all_vehicles, demand_file_path) == false) {
		return false;
	}

	return true;
}

bool initilizeCPU() {
	simulation_start_time = START_TIME_STEPS;
	simulation_end_time = END_TIME_STEPS; // 2 hours
	simulation_time_step = UNIT_TIME_STEPS;

	assert(simulation_time_step == 1);

	to_simulate_time = simulation_start_time;
	to_output_simulation_result_time = simulation_start_time;

//	roadThreadsInABlock = 32;
//	nodeThreadsInABlock = 32;
//	roadThreadsInABlock = 128;
//	nodeThreadsInABlock = 128;
	roadThreadsInABlock = 192;
	nodeThreadsInABlock = 192;

	roadBlocks = LANE_SIZE / roadThreadsInABlock + 1;
	nodeBlocks = NODE_SIZE / nodeThreadsInABlock + 1;

	simulation_results_pool.clear();
	simulation_results_output_file.open(simulation_output_file_path.c_str());
	simulation_results_output_file << "##TIME STEP" << ":Lane ID:" << ":(" << "COUNTS" << ":" << "flow" << ":" << "density" << ":" << "speed" << ":" << "queue_length" << ")" << endl;
	str_tools = new StringTools();

	return true;
}

__global__ void linkGPUData(GPUMemory *gpu_data, GPUVehicle *vpool_gpu, int *vpool_gpu_index, GPUSharedParameter* data_setting_gpu) {
	int time_index = threadIdx.x;
//	printf("time_index: %d\n", time_index);

	int nVehiclePerTick = data_setting_gpu->ON_GPU_VEHICLE_MAX_LOADING_ONE_TIME * data_setting_gpu->ON_GPU_LANE_SIZE;

//	printf("START Rebuild Vehicles on GPU\n");
	int counts = 0;
//	int sum = 0;

//	for (int time_index = 0; time_index < TOTAL_TIME_STEPS; time_index++) {
	for (int i = 0; i < data_setting_gpu->ON_GPU_LANE_SIZE; i++) {
		for (int j = 0; j < data_setting_gpu->ON_GPU_VEHICLE_MAX_LOADING_ONE_TIME; j++) {

			int index_t = time_index * nVehiclePerTick + i * data_setting_gpu->ON_GPU_VEHICLE_MAX_LOADING_ONE_TIME + j;

//				printf("START Rebuild Vehicles on GPU Done : %d vehciles rebuild\n", counts);

			int index_vehicle = vpool_gpu_index[index_t];

			if (index_vehicle >= 0) {
				gpu_data->new_vehicles_every_time_step[time_index].new_vehicles[i][j] = &(vpool_gpu[index_vehicle]);
				counts++;
//				printf("START Rebuild Vehicles on GPU Done : %d vehciles rebuild\n", counts);
			}
			else {
				gpu_data->new_vehicles_every_time_step[time_index].new_vehicles[i][j] = NULL;
//					printf("START Rebuild Vehicles on GPU Done : %d vehciles %d rebuild\n", counts, sum);
			}
		}
	}
//	}

//	printf("DDDD START Rebuild Vehicles on GPU Done : %d vehciles in %d rebuild\n", counts, sum);

//	GPUVehicle ***v = (GPUVehicle***) gpu_data->new_vehicles_every_time_step->new_vehicles;
}

bool initilizeGPU() {
	gpu_data = NULL;
	parameter_seeting_on_gpu = NULL;

	GPUMemory* data_local = new GPUMemory();
	initGPUData(data_local);

	GPUSharedParameter* data_setting_gpu = new GPUSharedParameter();
	initGPUParameterSetting(data_setting_gpu);

#ifdef ENABLE_CONSTANT
	GPUSharedParameter data_setting_cpu_constant;
	initGPUParameterSetting(&data_setting_cpu_constant);
#endif

	//apply memory on GPU
	size_t memory_space_for_vehicles = all_vehicles.size() * sizeof(GPUVehicle);
	if (cudaMalloc((void**) &vpool_gpu, memory_space_for_vehicles) != cudaSuccess) {
		cerr << "cudaMalloc((void**) &vpool_gpu, memory_space_for_vehicles) failed" << endl;
	}

	size_t memory_space_for_rebuild_index = TOTAL_TIME_STEPS * LANE_SIZE * VEHICLE_MAX_LOADING_ONE_TIME * sizeof(int);
	if (cudaMalloc((void**) &vpool_gpu_index, memory_space_for_rebuild_index) != cudaSuccess) {
		cerr << "cudaMalloc((void**) &vpool_gpu_index, memory_space_for_rebuild_index) failed" << endl;
	}

	if (cudaMalloc((void**) &gpu_data, data_local->total_size()) != cudaSuccess) {
		cerr << "cudaMalloc(&gpu_data, sizeof(GPUMemory)) failed" << endl;
	}

	if (cudaMalloc((void**) &parameter_seeting_on_gpu, sizeof(GPUSharedParameter)) != cudaSuccess) {
		cerr << "cudaMalloc(&GPUSharedParameter, sizeof(GPUSharedParameter)) failed" << endl;
	}

#ifdef ENABLE_CONSTANT
	cudaMemcpyToSymbol(data_setting_gpu_constant, &data_setting_cpu_constant, sizeof(GPUSharedParameter));
#endif

	//apply a buffer space for GPU outputs
	if (GPU_TO_CPU_SIMULATION_RESULTS_COPY_BUFFER_SIZE > 1) {
		size_t memory_space_for_buffer_outputs = sizeof(SimulationResults) * GPU_TO_CPU_SIMULATION_RESULTS_COPY_BUFFER_SIZE;
		if (cudaMalloc((void**) &simulation_results_buffer_on_gpu, memory_space_for_buffer_outputs) != cudaSuccess) {
			cerr << "cudaMalloc((void**) &simulation_results_buffer_on_gpu, memory_space_for_buffer_outputs) failed" << endl;
		}
	}

	cudaMemcpy(vpool_gpu, vpool_cpu, memory_space_for_vehicles, cudaMemcpyHostToDevice);
	cudaMemcpy(vpool_gpu_index, vpool_cpu_index, memory_space_for_rebuild_index, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_data, data_local, data_local->total_size(), cudaMemcpyHostToDevice);
	cudaMemcpy(parameter_seeting_on_gpu, data_setting_gpu, sizeof(GPUSharedParameter), cudaMemcpyHostToDevice);

	int GRID_SIZE = TOTAL_TIME_STEPS / 192;
	int BLOCK_SIZE = 192;
//	int BLOCK_SIZE = 1;

//	std::cout << "linkGPUData starts" << std::endl;
//	cudaEvent_t GPU_memory_rebuild_done;
//	cudaEventCreate(&GPU_memory_rebuild_done);

	linkGPUData<<<GRID_SIZE, BLOCK_SIZE>>>(gpu_data, vpool_gpu, vpool_gpu_index, parameter_seeting_on_gpu);
//	cudaEventRecord(GPU_memory_rebuild_done);
//	cudaStreamWaitEvent(NULL, GPU_memory_rebuild_done, 0);

	//try as a block
//	cudaMemcpy(data_local, gpu_data, data_local->total_size(), cudaMemcpyDeviceToHost);

	//wait for all CUDA related operations to finish;
	std::cout << "linkGPUData begins" << std::endl;
	cudaDeviceSynchronize();
	std::cout << "linkGPUData ends" << std::endl;

#ifdef ENABLE_OUTPUT_GPU_BUFFER
	cudaMallocHost((void **)&one_buffer, sizeof(SimulationResults) * GPU_TO_CPU_SIMULATION_RESULTS_COPY_BUFFER_SIZE);
#endif

	return true;
}

/*
 * Copy the parameter setting to GPU memory
 */
bool initGPUParameterSetting(GPUSharedParameter* data_setting_gpu) {

	data_setting_gpu->ON_GPU_Alpha = Alpha;
	data_setting_gpu->ON_GPU_Beta = Beta;
	data_setting_gpu->ON_GPU_END_TIME_STEPS = END_TIME_STEPS;
	data_setting_gpu->ON_GPU_GPU_TO_CPU_SIMULATION_RESULTS_COPY_BUFFER_SIZE = GPU_TO_CPU_SIMULATION_RESULTS_COPY_BUFFER_SIZE;
	data_setting_gpu->ON_GPU_LANE_INPUT_CAPACITY_TIME_STEP = LANE_INPUT_CAPACITY_TIME_STEP;
	data_setting_gpu->ON_GPU_LANE_OUTPUT_CAPACITY_TIME_STEP = LANE_OUTPUT_CAPACITY_TIME_STEP;
	data_setting_gpu->ON_GPU_LANE_SIZE = LANE_SIZE;
	data_setting_gpu->ON_GPU_MAXIMUM_LANE_CODING_LENGTH = MAXIMUM_LANE_CODING_LENGTH;
	data_setting_gpu->ON_GPU_MAX_LANE_DOWNSTREAM = MAX_LANE_DOWNSTREAM;
	data_setting_gpu->ON_GPU_MAX_LANE_UPSTREAM = MAX_LANE_UPSTREAM;
	data_setting_gpu->ON_GPU_MAX_ROUTE_LENGTH = MAX_ROUTE_LENGTH;
	data_setting_gpu->ON_GPU_MAX_SPEED = MAX_SPEED;

	data_setting_gpu->ON_GPU_MAX_VEHICLE_PER_LANE = MAX_VEHICLE_PER_LANE;
	data_setting_gpu->ON_GPU_MIN_SPEED = MIN_SPEED;
	data_setting_gpu->ON_GPU_Max_Density = Max_Density;
	data_setting_gpu->ON_GPU_Min_Density = Min_Density;
	data_setting_gpu->ON_GPU_NODE_SIZE = NODE_SIZE;
	data_setting_gpu->ON_GPU_ROAD_LENGTH = ROAD_LENGTH;
	data_setting_gpu->ON_GPU_START_TIME_STEPS = START_TIME_STEPS;
	data_setting_gpu->ON_GPU_TOTAL_TIME_STEPS = TOTAL_TIME_STEPS;
	data_setting_gpu->ON_GPU_UNIT_TIME_STEPS = UNIT_TIME_STEPS;
	data_setting_gpu->ON_GPU_VEHICLE_LENGTH = VEHICLE_LENGTH;
	data_setting_gpu->ON_GPU_VEHICLE_MAX_LOADING_ONE_TIME = VEHICLE_MAX_LOADING_ONE_TIME;

	return true;
}

/*
 * Build a GPU data
 */
bool initGPUData(GPUMemory* data_local) {

	/**
	 * First Part: Lane
	 */

	for (int i = 0; i < the_network->all_links.size(); i++) {
		Link* one_link = the_network->all_links[i];

		data_local->lane_pool.lane_ID[i] = one_link->link_id;
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
		data_local->lane_pool.lane_length[i] = ROAD_LENGTH; // meter
		data_local->lane_pool.max_vehicles[i] = ROAD_LENGTH / VEHICLE_LENGTH; //number of vehicles
		data_local->lane_pool.output_capacity[i] = LANE_OUTPUT_CAPACITY_TIME_STEP; //
		data_local->lane_pool.input_capacity[i] = LANE_INPUT_CAPACITY_TIME_STEP; //
		data_local->lane_pool.empty_space[i] = ROAD_LENGTH;

		/*
		 * for speed calculation
		 */
		data_local->lane_pool.alpha[i] = Alpha;
		data_local->lane_pool.beta[i] = Beta;
		data_local->lane_pool.max_density[i] = Max_Density;
		data_local->lane_pool.min_density[i] = Min_Density;
		data_local->lane_pool.MAX_SPEED[i] = MAX_SPEED;
		data_local->lane_pool.MIN_SPEED[i] = MIN_SPEED;

		data_local->lane_pool.vehicle_counts[i] = 0;
		data_local->lane_pool.vehicle_passed_to_the_lane_counts[i] = 0;

		for (int c = 0; c < MAX_VEHICLE_PER_LANE; c++) {
			data_local->lane_pool.vehicle_space[c][i] = NULL;
		}

		for (int c = 0; c < LANE_INPUT_CAPACITY_TIME_STEP; c++) {
			data_local->lane_pool.vehicle_passed_space[c][i] = NULL;
		}

		for (int j = 0; j < TOTAL_TIME_STEPS; j++) {
			data_local->lane_pool.speed_history[j][i] = -1;
		}

		//it is assumed that QUEUE_LENGTH_HISTORY = 4;
		assert(QUEUE_LENGTH_HISTORY == 4);
		float weight[QUEUE_LENGTH_HISTORY];
		weight[0] = 1.0;
		weight[1] = 0;
		weight[2] = 0;
		weight[3] = 0;

		//		{ 0.2, 0.3, 0.5, 0 };

		for (int j = 0; j < QUEUE_LENGTH_HISTORY; j++) {
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

		data_local->node_pool.node_ID[i] = one_node->node_id;
		data_local->node_pool.MAXIMUM_ACCUMULATED_FLOW[i] = 0;
		data_local->node_pool.ACCUMULATYED_UPSTREAM_CAPACITY[i] = 0;
		data_local->node_pool.ACCUMULATYED_DOWNSTREAM_CAPACITY[i] = 0;

//		assert(one_node->node_id == i);

		for (int j = 0; j < MAX_LANE_UPSTREAM; j++) {
			data_local->node_pool.upstream[j][i] = -1;
		}

		for (int j = 0; j < one_node->upstream_links.size(); j++) {
			data_local->node_pool.upstream[j][i] = one_node->upstream_links[j]->link_id;
			data_local->node_pool.ACCUMULATYED_UPSTREAM_CAPACITY[i] += LANE_OUTPUT_CAPACITY_TIME_STEP;
		}

		for (int j = 0; j < MAX_LANE_DOWNSTREAM; j++) {
			data_local->node_pool.downstream[j][i] = -1;
		}

		for (int j = 0; j < one_node->downstream_links.size(); j++) {
			data_local->node_pool.downstream[j][i] = one_node->downstream_links[j]->link_id;
			data_local->node_pool.ACCUMULATYED_DOWNSTREAM_CAPACITY[i] += LANE_OUTPUT_CAPACITY_TIME_STEP;
		}

		if (data_local->node_pool.ACCUMULATYED_UPSTREAM_CAPACITY[i] <= 0) {
			data_local->node_pool.MAXIMUM_ACCUMULATED_FLOW[i] = data_local->node_pool.ACCUMULATYED_DOWNSTREAM_CAPACITY[i];
		}
		else if (data_local->node_pool.ACCUMULATYED_DOWNSTREAM_CAPACITY[i] <= 0) {
			data_local->node_pool.MAXIMUM_ACCUMULATED_FLOW[i] = data_local->node_pool.ACCUMULATYED_UPSTREAM_CAPACITY[i];
		}
		else {
			data_local->node_pool.MAXIMUM_ACCUMULATED_FLOW[i] = std::min(data_local->node_pool.ACCUMULATYED_UPSTREAM_CAPACITY[i], data_local->node_pool.ACCUMULATYED_DOWNSTREAM_CAPACITY[i]);
		}

//		std::cout << "MAXIMUM_ACCUMULATED_FLOW:" << i << ", " << data_local->node_pool.MAXIMUM_ACCUMULATED_FLOW[i] << std::endl;
	}

	/**
	 * Third Part:
	 */

//Init VehiclePool
	for (int i = START_TIME_STEPS; i < TOTAL_TIME_STEPS; i += UNIT_TIME_STEPS) {
		for (int j = 0; j < LANE_SIZE; j++) {
			data_local->new_vehicles_every_time_step[i].new_vehicle_size[j] = 0;
			data_local->new_vehicles_every_time_step[i].lane_ID[j] = -1;
		}
	}

	std::cout << "all_vehicles.size():" << all_vehicles.size() << std::endl;

//init host vehicle pool data /*xiaosong*/
	int memory_space_for_vehicles = all_vehicles.size() * sizeof(GPUVehicle);
	vpool_cpu = (GPUVehicle*) malloc(memory_space_for_vehicles);
	if (vpool_cpu == NULL) exit(1);

	std::cout << "vpool_cpu_index size:" << TOTAL_TIME_STEPS * LANE_SIZE * VEHICLE_MAX_LOADING_ONE_TIME << std::endl;
	vpool_cpu_index = (int*) malloc(TOTAL_TIME_STEPS * LANE_SIZE * VEHICLE_MAX_LOADING_ONE_TIME * sizeof(int));
	if (vpool_cpu_index == NULL) exit(1);

	for (int i = START_TIME_STEPS; i < TOTAL_TIME_STEPS; i += UNIT_TIME_STEPS) {
		for (int j = 0; j < LANE_SIZE; j++) {
			for (int z = 0; z < VEHICLE_MAX_LOADING_ONE_TIME; z++) {

				int index_t = i * LANE_SIZE * VEHICLE_MAX_LOADING_ONE_TIME + j * VEHICLE_MAX_LOADING_ONE_TIME + z;

				//-1 means there is no vehicle on the road
				vpool_cpu_index[index_t] = -1;
			}
		}
	}

	int nVehiclePerTick = VEHICLE_MAX_LOADING_ONE_TIME * LANE_SIZE;

//	std::cout << "total array size:" << (TOTAL_TIME_STEPS * LANE_SIZE * VEHICLE_MAX_LOADING_ONE_TIME) << std::endl;
//	std::cout << "total size:" << (TOTAL_TIME_STEPS * LANE_SIZE * VEHICLE_MAX_LOADING_ONE_TIME * sizeof(GPUVehicle)) << std::endl;

	std::cout << "init all_vehicles" << std::endl;

	int total_inserted_vehicles = 0;

//Insert Vehicles
	for (int i = 0; i < all_vehicles.size(); i++) {
		Vehicle* one_vehicle = all_vehicles[i];

		int time_index = one_vehicle->entry_time;
		int time_index_covert = timestep_to_arrayindex(time_index);
		assert(time_index == time_index_covert);
		//try to load vehicles beyond the simulation border
		if (time_index_covert >= TOTAL_TIME_STEPS) continue;

		int lane_ID = all_od_paths[one_vehicle->path_id]->link_ids[0];

		if (data_local->new_vehicles_every_time_step[time_index_covert].new_vehicle_size[lane_ID] < VEHICLE_MAX_LOADING_ONE_TIME) {
			int last_vehicle_index = data_local->new_vehicles_every_time_step[time_index_covert].new_vehicle_size[lane_ID];
			int idx_vpool = time_index_covert * nVehiclePerTick + lane_ID * VEHICLE_MAX_LOADING_ONE_TIME + last_vehicle_index;

			//for gpu to rebuild
			vpool_cpu_index[idx_vpool] = i;
//			std::cout << "idx_vpool:" << idx_vpool << " is map to i:" << i << std::endl;

			vpool_cpu[i].vehicle_ID = one_vehicle->vehicle_id;
			vpool_cpu[i].entry_time = time_index;
			vpool_cpu[i].current_lane_ID = lane_ID;
			int max_copy_length = MAX_ROUTE_LENGTH > all_od_paths[one_vehicle->path_id]->link_ids.size() ? all_od_paths[one_vehicle->path_id]->link_ids.size() : MAX_ROUTE_LENGTH;

			for (int p = 0; p < max_copy_length; p++) {
				vpool_cpu[i].path_code[p] = all_od_paths[one_vehicle->path_id]->route_code[p] ? 1 : 0;
			}

			//ready for the next lane, so next_path_index is set to 1, if the next_path_index == whole_path_length, it means cannot find path any more, can exit;
			vpool_cpu[i].next_path_index = 1;
			vpool_cpu[i].whole_path_length = all_od_paths[one_vehicle->path_id]->link_ids.size();

			//will be re-writen by GPU
			data_local->new_vehicles_every_time_step[time_index_covert].new_vehicles[lane_ID][last_vehicle_index] = &(vpool_cpu[i]);
			data_local->new_vehicles_every_time_step[time_index_covert].new_vehicle_size[lane_ID]++;

			total_inserted_vehicles++;
		}
		else {
			std::cout << "Loading Vehicles Exceeds The Loading Capacity: Time:" << time_index_covert << ", Lane_ID:" << lane_ID << ",i:" << i << ",ID:" << one_vehicle->vehicle_id << std::endl;
		}
	}

	std::cout << "init all_vehicles done:" << total_inserted_vehicles << std::endl;

	return true;
}

bool destory_resources() {
	simulation_results_output_file.flush();
	simulation_results_output_file.close();

//	cudaEventDestroy(GPU_supply_one_time_simulation_done_event);
//	cudaStreamDestroy(stream_gpu_supply);

	if (vpool_cpu != NULL) delete vpool_cpu;
//	if (vpool_cpu_index != NULL) delete vpool_cpu_index;
	if (str_tools != NULL) delete str_tools;
	cudaDeviceReset();
	return true;
}

bool start_simulation() {
	bool first_time_step = true;

	/*
	 * Simulation Loop
	 */

	while (((to_simulate_time >= simulation_end_time) && (to_output_simulation_result_time >= simulation_end_time)) == false) {

		//GPU has done simulation at current time
		if (to_simulate_time < simulation_end_time && (cudaEventQuery(GPU_supply_one_time_simulation_done_event) == cudaSuccess)) {
			//step 1
//			cout << "cudaEventQuery return true, to_simulate_time:" << to_simulate_time << endl;

			if (first_time_step == true) {
				first_time_step = false;
			}
			else {
#ifdef ENABLE_OUTPUT_GPU_BUFFER
				if ((to_simulate_time + 1) % GPU_TO_CPU_SIMULATION_RESULTS_COPY_BUFFER_SIZE == 0) {
					copy_buffer_simulated_results_to_CPU(to_simulate_time);
				}
#else
				copy_simulated_results_to_CPU(to_simulate_time);
#endif

				to_simulate_time += simulation_time_step;
			}

			if (to_simulate_time < simulation_end_time) {
				//step 2
#ifdef ENABLE_OUTPUT
//				cout << "to_simulate_time:" << to_simulate_time << ", simulation_end_time:" << simulation_end_time << endl;
#endif
//				cout << "to_simulate_time:" << to_simulate_time << ", simulation_end_time:" << simulation_end_time << endl;

				//setp 3
				supply_simulation_pre_vehicle_passing<<<roadBlocks, roadThreadsInABlock, 0, stream_gpu_supply>>>(gpu_data, to_simulate_time, LANE_SIZE, parameter_seeting_on_gpu);
//			cout << "supply_simulation_pre_vehicle_passing done" << endl;

				supply_simulation_vehicle_passing<<<nodeBlocks, nodeThreadsInABlock, 0, stream_gpu_supply>>>(gpu_data, to_simulate_time, NODE_SIZE, parameter_seeting_on_gpu);
//			cout << "supply_simulation_vehicle_passing done" << endl;

				supply_simulation_after_vehicle_passing<<<roadBlocks, roadThreadsInABlock, 0, stream_gpu_supply>>>(gpu_data, to_simulate_time, LANE_SIZE, parameter_seeting_on_gpu);
//			cout << "supply_simulation_after_vehicle_passing done" << endl;

#ifdef ENABLE_OUTPUT_GPU_BUFFER
				supply_simulated_results_to_buffer<<<roadBlocks, roadThreadsInABlock, 0, stream_gpu_supply>>>(gpu_data, to_simulate_time, LANE_SIZE, simulation_results_buffer_on_gpu,
						parameter_seeting_on_gpu);
#endif

				cudaEventRecord(GPU_supply_one_time_simulation_done_event, stream_gpu_supply);
			}
//			cout << "cudaEventRecord done" << endl;
		}
		//GPU is busy, so CPU does something else (I/O)
		else if (to_output_simulation_result_time < to_simulate_time) {

#ifdef ENABLE_OUTPUT_GPU_BUFFER
			if (to_output_simulation_result_time <= to_simulate_time - GPU_TO_CPU_SIMULATION_RESULTS_COPY_BUFFER_SIZE) {

#ifdef ENABLE_OUTPUT
				output_buffered_simulated_results(to_output_simulation_result_time);
#endif
				to_output_simulation_result_time += simulation_time_step * GPU_TO_CPU_SIMULATION_RESULTS_COPY_BUFFER_SIZE;
			}
#else
#ifdef ENABLE_OUTPUT
			output_simulated_results(to_output_simulation_result_time);
#endif
			to_output_simulation_result_time += simulation_time_step;
#endif
		}
		else {

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
bool copy_simulated_results_to_CPU(int time_step) {
	int index = timestep_to_arrayindex(time_step);
	SimulationResults* one = new SimulationResults();

//	cout << "copy_simulated_results_to_CPU starts" << endl;

	cudaMemcpy(one->flow, gpu_data->lane_pool.flow, sizeof(float) * LANE_SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(one->density, gpu_data->lane_pool.density, sizeof(float) * LANE_SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(one->speed, gpu_data->lane_pool.speed, sizeof(float) * LANE_SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(one->queue_length, gpu_data->lane_pool.queue_length, sizeof(float) * LANE_SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(one->counts, gpu_data->lane_pool.vehicle_counts, sizeof(int) * LANE_SIZE, cudaMemcpyDeviceToHost);

//	cout << "copy_simulated_results_to_CPU ends" << endl;

	simulation_results_pool[index] = one;

//	cout << "copy_simulated_results_to_CPU done" << endl;
	return true;
}

bool copy_buffer_simulated_results_to_CPU(int time_step) {
//	std::cout << "copy_buffer_simulated_results_to_CPU AT time " << time_step << std::endl;
//	SimulationResults* one_buffer = (SimulationResults*) malloc(sizeof(SimulationResults) * GPU_TO_CPU_SIMULATION_RESULTS_COPY_BUFFER_SIZE);
//	cudaMemcpy(one_buffer, simulation_results_buffer_on_gpu, sizeof(SimulationResults) * GPU_TO_CPU_SIMULATION_RESULTS_COPY_BUFFER_SIZE, cudaMemcpyDeviceToHost);

	cudaMemcpyAsync(one_buffer, simulation_results_buffer_on_gpu, sizeof(SimulationResults) * GPU_TO_CPU_SIMULATION_RESULTS_COPY_BUFFER_SIZE, cudaMemcpyDeviceToHost, stream_gpu_io);

	for (int i = 0; i < GPU_TO_CPU_SIMULATION_RESULTS_COPY_BUFFER_SIZE; i++) {
		int time_index = time_step - (GPU_TO_CPU_SIMULATION_RESULTS_COPY_BUFFER_SIZE - 1) + i;
		simulation_results_pool[time_index] = &one_buffer[i];
	}

	return true;
}

bool output_simulated_results(int time_step) {
	if (simulation_results_pool.find(time_step) == simulation_results_pool.end()) {
		std::cerr << "System Error, Try to output time " << time_step << ", while it is not ready!" << std::endl;
		return false;
	}

	int index = timestep_to_arrayindex(time_step);
	SimulationResults* one = simulation_results_pool[index];
	assert(one != NULL);

	for (int i = 0; i < LANE_SIZE; i++) {
		simulation_results_output_file << time_step << ":lane:" << i << ":(" << one->counts[i] << ":" << one->flow[i] << ":" << one->density[i] << ":" << one->speed[i] << ":" << one->queue_length[i]
				<< ")" << endl;
	}

//	temply not deleted
//	if(one != NULL)
//		delete one;

	return true;
}

bool output_buffered_simulated_results(int time_step) {
	std::cout << "output_buffered_simulated_results AT time " << time_step << std::endl;

	for (int i = 0; i < GPU_TO_CPU_SIMULATION_RESULTS_COPY_BUFFER_SIZE; i++) {
		output_simulated_results(time_step + i);
	}

	return true;
}

/**
 * Kernel Functions, not sure how to move to other folder
 */

#ifdef ENABLE_MORE_REGISTER
/*
 * Supply Function Implementation
 */__global__ void supply_simulation_pre_vehicle_passing(GPUMemory* gpu_data, int time_step, int segment_length, GPUSharedParameter* data_setting_gpu) {
	int lane_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (lane_id >= segment_length) return;

	int time_index = time_step;

//	gpu_data->lane_pool.new_vehicle_join_counts[lane_id] = 0;

//init capacity
#ifdef ENABLE_CONSTANT
	gpu_data->lane_pool.input_capacity[lane_id] = data_setting_gpu_constant.ON_GPU_LANE_INPUT_CAPACITY_TIME_STEP;
	gpu_data->lane_pool.output_capacity[lane_id] = data_setting_gpu_constant.ON_GPU_LANE_OUTPUT_CAPACITY_TIME_STEP;
#else
	gpu_data->lane_pool.input_capacity[lane_id] = data_setting_gpu->ON_GPU_LANE_INPUT_CAPACITY_TIME_STEP;
	gpu_data->lane_pool.output_capacity[lane_id] = data_setting_gpu->ON_GPU_LANE_OUTPUT_CAPACITY_TIME_STEP;
#endif

//init for next GPU kernel function
	gpu_data->lane_pool.blocked[lane_id] = false;

//load passed vehicles to the back of the lane
	for (int i = 0; i < gpu_data->lane_pool.vehicle_passed_to_the_lane_counts[lane_id]; i++) {
		if (gpu_data->lane_pool.vehicle_counts[lane_id] < gpu_data->lane_pool.max_vehicles[lane_id]) {
			gpu_data->lane_pool.vehicle_space[gpu_data->lane_pool.vehicle_counts[lane_id]][lane_id] = gpu_data->lane_pool.vehicle_passed_space[i][lane_id];
			gpu_data->lane_pool.vehicle_counts[lane_id]++;

//				gpu_data->lane_pool.new_vehicle_join_counts[lane_id]++;
		}
	}

	if (gpu_data->lane_pool.vehicle_passed_to_the_lane_counts[lane_id] > 0) {
#ifdef ENABLE_CONSTANT
		gpu_data->lane_pool.empty_space[lane_id] = min_device(gpu_data->lane_pool.speed[lane_id] * data_setting_gpu_constant.ON_GPU_UNIT_TIME_STEPS, gpu_data->lane_pool.empty_space[lane_id])
				- gpu_data->lane_pool.vehicle_passed_to_the_lane_counts[lane_id] * data_setting_gpu_constant.ON_GPU_VEHICLE_LENGTH;
#else
		gpu_data->lane_pool.empty_space[lane_id] = min_device(gpu_data->lane_pool.speed[lane_id] * data_setting_gpu->ON_GPU_UNIT_TIME_STEPS, gpu_data->lane_pool.empty_space[lane_id])
		- gpu_data->lane_pool.vehicle_passed_to_the_lane_counts[lane_id] * data_setting_gpu->ON_GPU_VEHICLE_LENGTH;
#endif
		if (gpu_data->lane_pool.empty_space[lane_id] < 0) gpu_data->lane_pool.empty_space[lane_id] = 0;
	}

	gpu_data->lane_pool.last_time_empty_space[lane_id] = gpu_data->lane_pool.empty_space[lane_id];
	gpu_data->lane_pool.vehicle_passed_to_the_lane_counts[lane_id] = 0;

//
//load newly generated vehicles to the back of the lane
	for (int i = 0; i < gpu_data->new_vehicles_every_time_step[time_index].new_vehicle_size[lane_id]; i++) {
		if (gpu_data->lane_pool.vehicle_counts[lane_id] < gpu_data->lane_pool.max_vehicles[lane_id]) {

			gpu_data->lane_pool.vehicle_space[gpu_data->lane_pool.vehicle_counts[lane_id]][lane_id] = (gpu_data->new_vehicles_every_time_step[time_index].new_vehicles[lane_id][i]);
			gpu_data->lane_pool.vehicle_counts[lane_id]++;

//			gpu_data->lane_pool.new_vehicle_join_counts[lane_id]++;
		}
	}

	float density_ = 0.0f;
	float speed_ = 0.0f;

//update speed and density
#ifdef ENABLE_CONSTANT
	density_ = 1.0 * data_setting_gpu_constant.ON_GPU_VEHICLE_LENGTH * gpu_data->lane_pool.vehicle_counts[lane_id] / data_setting_gpu_constant.ON_GPU_ROAD_LENGTH;
#else
	density_ = 1.0 * data_setting_gpu->ON_GPU_VEHICLE_LENGTH * gpu_data->lane_pool.vehicle_counts[lane_id] / gpu_data->lane_pool.lane_length[lane_id];
#endif

//Speed-Density Relationship
	if (density_ < data_setting_gpu_constant.ON_GPU_Min_Density) speed_ = data_setting_gpu_constant.ON_GPU_MAX_SPEED;
	else {
		speed_ = data_setting_gpu_constant.ON_GPU_MAX_SPEED
				- data_setting_gpu_constant.ON_GPU_MAX_SPEED / (data_setting_gpu_constant.ON_GPU_Max_Density - data_setting_gpu_constant.ON_GPU_Min_Density)
						* (density_ - data_setting_gpu_constant.ON_GPU_Min_Density);
	}
//		gpu_data->lane_pool.speed[lane_id] = ( gpu_data->lane_pool.MAX_SPEED[lane_id] - gpu_data->lane_pool.MIN_SPEED ) / gpu_data->lane_pool.max_density[lane_id] * ( gpu_data->lane_pool.max_density[lane_id] - 0 );

	if (speed_ < data_setting_gpu_constant.ON_GPU_MIN_SPEED) speed_ = data_setting_gpu_constant.ON_GPU_MIN_SPEED;

//update speed history
	gpu_data->lane_pool.speed_history[time_index][lane_id] = speed_;

	gpu_data->lane_pool.density[lane_id] = density_;
	gpu_data->lane_pool.speed[lane_id] = speed_;
//estimated empty_space

	float prediction_queue_length_ = 0.0f;

	if (time_step < START_TIME_STEPS + 4 * UNIT_TIME_STEPS) {
//		gpu_data->lane_pool.predicted_empty_space[lane_id] = gpu_data->lane_pool.his_queue_length[0][lane_id];
//		gpu_data->lane_pool.predicted_queue_length[lane_id] = 0;

#ifdef ENABLE_CONSTANT
		gpu_data->lane_pool.predicted_empty_space[lane_id] = min_device(gpu_data->lane_pool.last_time_empty_space[lane_id] + (speed_ * data_setting_gpu_constant.ON_GPU_UNIT_TIME_STEPS),
				1.0f * data_setting_gpu_constant.ON_GPU_ROAD_LENGTH);
#else
		gpu_data->lane_pool.predicted_empty_space[lane_id] = min_device(
				gpu_data->lane_pool.last_time_empty_space[lane_id] + (gpu_data->lane_pool.speed[lane_id] * data_setting_gpu->ON_GPU_UNIT_TIME_STEPS), 1.0f * data_setting_gpu->ON_GPU_ROAD_LENGTH);
#endif
	}
	else {
		prediction_queue_length_ = gpu_data->lane_pool.his_queue_length[0][lane_id];
		prediction_queue_length_ += (gpu_data->lane_pool.his_queue_length[0][lane_id] - gpu_data->lane_pool.his_queue_length[1][lane_id]) * gpu_data->lane_pool.his_queue_length_weighting[0][lane_id];

//		prediction_empty_space_ += (gpu_data->lane_pool.his_queue_length[1][lane_id] - gpu_data->lane_pool.his_queue_length[2][lane_id])
//				* gpu_data->lane_pool.his_queue_length_weighting[1][lane_id];
//
//		prediction_empty_space_ += (gpu_data->lane_pool.his_queue_length[2][lane_id] - gpu_data->lane_pool.his_queue_length[3][lane_id])
//				* gpu_data->lane_pool.his_queue_length_weighting[2][lane_id];

		//need improve
		//XUYAN, need modify
#ifdef ENABLE_CONSTANT
		gpu_data->lane_pool.predicted_empty_space[lane_id] = min_device(gpu_data->lane_pool.last_time_empty_space[lane_id] + (speed_ * data_setting_gpu_constant.ON_GPU_UNIT_TIME_STEPS),
				(data_setting_gpu_constant.ON_GPU_ROAD_LENGTH - prediction_queue_length_));
#else
		gpu_data->lane_pool.predicted_empty_space[lane_id] = min_device(
				gpu_data->lane_pool.last_time_empty_space[lane_id] + (gpu_data->lane_pool.speed[lane_id] * data_setting_gpu->ON_GPU_UNIT_TIME_STEPS),
				(data_setting_gpu->ON_GPU_ROAD_LENGTH - prediction_empty_space_));
#endif
	}

//	gpu_data->lane_pool.debug_data[lane_id] = gpu_data->lane_pool.predicted_empty_space[lane_id];
//update Tp

#ifdef ENABLE_CONSTANT
	gpu_data->lane_pool.accumulated_offset[lane_id] += speed_ * data_setting_gpu_constant.ON_GPU_UNIT_TIME_STEPS; //meter

	while (gpu_data->lane_pool.accumulated_offset[lane_id] >= data_setting_gpu_constant.ON_GPU_ROAD_LENGTH) {
		gpu_data->lane_pool.accumulated_offset[lane_id] -= gpu_data->lane_pool.speed_history[gpu_data->lane_pool.Tp[lane_id]][lane_id] * data_setting_gpu_constant.ON_GPU_UNIT_TIME_STEPS;
		gpu_data->lane_pool.Tp[lane_id] += data_setting_gpu_constant.ON_GPU_UNIT_TIME_STEPS;
	}
#else

	gpu_data->lane_pool.accumulated_offset[lane_id] += gpu_data->lane_pool.speed[lane_id] * data_setting_gpu->ON_GPU_UNIT_TIME_STEPS; //meter

	while (gpu_data->lane_pool.accumulated_offset[lane_id] >= gpu_data->lane_pool.lane_length[lane_id]) {
		gpu_data->lane_pool.accumulated_offset[lane_id] -= gpu_data->lane_pool.speed_history[gpu_data->lane_pool.Tp[lane_id]][lane_id] * data_setting_gpu->ON_GPU_UNIT_TIME_STEPS;
		gpu_data->lane_pool.Tp[lane_id] += data_setting_gpu->ON_GPU_UNIT_TIME_STEPS;
	}
#endif

	//update queue length
	int queue_start = gpu_data->lane_pool.queue_length[lane_id] / data_setting_gpu_constant.ON_GPU_VEHICLE_LENGTH;
	for (int queue_index = queue_start; queue_index < gpu_data->lane_pool.vehicle_counts[lane_id]; queue_index++) {
		if (gpu_data->lane_pool.vehicle_space[queue_index][lane_id]->entry_time <= gpu_data->lane_pool.Tp[lane_id]) {
			gpu_data->lane_pool.queue_length[lane_id] += data_setting_gpu_constant.ON_GPU_VEHICLE_LENGTH;
		}
		else {
			break;
		}
	}
}

#else
/*
 * Supply Function Implementation
 */__global__ void supply_simulation_pre_vehicle_passing(GPUMemory* gpu_data, int time_step, int segment_length, GPUSharedParameter* data_setting_gpu) {
	int lane_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (lane_id >= segment_length) return;

	int time_index = time_step;

//	gpu_data->lane_pool.new_vehicle_join_counts[lane_id] = 0;

//init capacity
#ifdef ENABLE_CONSTANT
	gpu_data->lane_pool.input_capacity[lane_id] = data_setting_gpu_constant.ON_GPU_LANE_INPUT_CAPACITY_TIME_STEP;
	gpu_data->lane_pool.output_capacity[lane_id] = data_setting_gpu_constant.ON_GPU_LANE_OUTPUT_CAPACITY_TIME_STEP;
#else
	gpu_data->lane_pool.input_capacity[lane_id] = data_setting_gpu->ON_GPU_LANE_INPUT_CAPACITY_TIME_STEP;
	gpu_data->lane_pool.output_capacity[lane_id] = data_setting_gpu->ON_GPU_LANE_OUTPUT_CAPACITY_TIME_STEP;
#endif

//init for next GPU kernel function
	gpu_data->lane_pool.blocked[lane_id] = false;

//load passed vehicles to the back of the lane
	for (int i = 0; i < gpu_data->lane_pool.vehicle_passed_to_the_lane_counts[lane_id]; i++) {
		if (gpu_data->lane_pool.vehicle_counts[lane_id] < gpu_data->lane_pool.max_vehicles[lane_id]) {
			gpu_data->lane_pool.vehicle_space[gpu_data->lane_pool.vehicle_counts[lane_id]][lane_id] = gpu_data->lane_pool.vehicle_passed_space[i][lane_id];
			gpu_data->lane_pool.vehicle_counts[lane_id]++;

//				gpu_data->lane_pool.new_vehicle_join_counts[lane_id]++;
		}
	}

	if (gpu_data->lane_pool.vehicle_passed_to_the_lane_counts[lane_id] > 0) {
#ifdef ENABLE_CONSTANT
		gpu_data->lane_pool.empty_space[lane_id] = min_device(gpu_data->lane_pool.speed[lane_id] * data_setting_gpu_constant.ON_GPU_UNIT_TIME_STEPS, gpu_data->lane_pool.empty_space[lane_id])
		- gpu_data->lane_pool.vehicle_passed_to_the_lane_counts[lane_id] * data_setting_gpu_constant.ON_GPU_VEHICLE_LENGTH;
#else
		gpu_data->lane_pool.empty_space[lane_id] = min_device(gpu_data->lane_pool.speed[lane_id] * data_setting_gpu->ON_GPU_UNIT_TIME_STEPS, gpu_data->lane_pool.empty_space[lane_id])
		- gpu_data->lane_pool.vehicle_passed_to_the_lane_counts[lane_id] * data_setting_gpu->ON_GPU_VEHICLE_LENGTH;
#endif
		if (gpu_data->lane_pool.empty_space[lane_id] < 0) gpu_data->lane_pool.empty_space[lane_id] = 0;
	}

	gpu_data->lane_pool.last_time_empty_space[lane_id] = gpu_data->lane_pool.empty_space[lane_id];
	gpu_data->lane_pool.vehicle_passed_to_the_lane_counts[lane_id] = 0;

//
//load newly generated vehicles to the back of the lane
	for (int i = 0; i < gpu_data->new_vehicles_every_time_step[time_index].new_vehicle_size[lane_id]; i++) {
		if (gpu_data->lane_pool.vehicle_counts[lane_id] < gpu_data->lane_pool.max_vehicles[lane_id]) {

			gpu_data->lane_pool.vehicle_space[gpu_data->lane_pool.vehicle_counts[lane_id]][lane_id] = (gpu_data->new_vehicles_every_time_step[time_index].new_vehicles[lane_id][i]);
			gpu_data->lane_pool.vehicle_counts[lane_id]++;

//			gpu_data->lane_pool.new_vehicle_join_counts[lane_id]++;
		}
	}

//update speed and density
#ifdef ENABLE_CONSTANT
	gpu_data->lane_pool.density[lane_id] = 1.0 * data_setting_gpu_constant.ON_GPU_VEHICLE_LENGTH * gpu_data->lane_pool.vehicle_counts[lane_id] / gpu_data->lane_pool.lane_length[lane_id];
#else
	gpu_data->lane_pool.density[lane_id] = 1.0 * data_setting_gpu->ON_GPU_VEHICLE_LENGTH * gpu_data->lane_pool.vehicle_counts[lane_id] / gpu_data->lane_pool.lane_length[lane_id];
#endif

//Speed-Density Relationship
	if (gpu_data->lane_pool.density[lane_id] < gpu_data->lane_pool.min_density[lane_id]) gpu_data->lane_pool.speed[lane_id] = gpu_data->lane_pool.MAX_SPEED[lane_id];
	else {
		gpu_data->lane_pool.speed[lane_id] = gpu_data->lane_pool.MAX_SPEED[lane_id]
		- gpu_data->lane_pool.MAX_SPEED[lane_id] / (gpu_data->lane_pool.max_density[lane_id] - gpu_data->lane_pool.min_density[lane_id])
		* (gpu_data->lane_pool.density[lane_id] - gpu_data->lane_pool.min_density[lane_id]);
	}
//		gpu_data->lane_pool.speed[lane_id] = ( gpu_data->lane_pool.MAX_SPEED[lane_id] - gpu_data->lane_pool.MIN_SPEED ) / gpu_data->lane_pool.max_density[lane_id] * ( gpu_data->lane_pool.max_density[lane_id] - 0 );

	if (gpu_data->lane_pool.speed[lane_id] < gpu_data->lane_pool.MIN_SPEED[lane_id]) gpu_data->lane_pool.speed[lane_id] = gpu_data->lane_pool.MIN_SPEED[lane_id];

//update speed history
	gpu_data->lane_pool.speed_history[time_index][lane_id] = gpu_data->lane_pool.speed[lane_id];

//estimated empty_space

	if (time_step < START_TIME_STEPS + 4 * UNIT_TIME_STEPS) {
//		gpu_data->lane_pool.predicted_empty_space[lane_id] = gpu_data->lane_pool.his_queue_length[0][lane_id];
		gpu_data->lane_pool.predicted_queue_length[lane_id] = 0;

#ifdef ENABLE_CONSTANT
		gpu_data->lane_pool.predicted_empty_space[lane_id] = min_device(
				gpu_data->lane_pool.last_time_empty_space[lane_id] + (gpu_data->lane_pool.speed[lane_id] * data_setting_gpu_constant.ON_GPU_UNIT_TIME_STEPS),
				1.0f * data_setting_gpu_constant.ON_GPU_ROAD_LENGTH);
#else
		gpu_data->lane_pool.predicted_empty_space[lane_id] = min_device(
				gpu_data->lane_pool.last_time_empty_space[lane_id] + (gpu_data->lane_pool.speed[lane_id] * data_setting_gpu->ON_GPU_UNIT_TIME_STEPS), 1.0f * data_setting_gpu->ON_GPU_ROAD_LENGTH);
#endif
	}
	else {
		gpu_data->lane_pool.predicted_queue_length[lane_id] = gpu_data->lane_pool.his_queue_length[0][lane_id];
		gpu_data->lane_pool.predicted_queue_length[lane_id] += (gpu_data->lane_pool.his_queue_length[0][lane_id] - gpu_data->lane_pool.his_queue_length[1][lane_id])
		* gpu_data->lane_pool.his_queue_length_weighting[0][lane_id];

		gpu_data->lane_pool.predicted_queue_length[lane_id] += (gpu_data->lane_pool.his_queue_length[1][lane_id] - gpu_data->lane_pool.his_queue_length[2][lane_id])
		* gpu_data->lane_pool.his_queue_length_weighting[1][lane_id];

		gpu_data->lane_pool.predicted_queue_length[lane_id] += (gpu_data->lane_pool.his_queue_length[2][lane_id] - gpu_data->lane_pool.his_queue_length[3][lane_id])
		* gpu_data->lane_pool.his_queue_length_weighting[2][lane_id];

		//need improve
		//XUYAN, need modify
#ifdef ENABLE_CONSTANT
		gpu_data->lane_pool.predicted_empty_space[lane_id] = min_device(
				gpu_data->lane_pool.last_time_empty_space[lane_id] + (gpu_data->lane_pool.speed[lane_id] * data_setting_gpu_constant.ON_GPU_UNIT_TIME_STEPS),
				(data_setting_gpu_constant.ON_GPU_ROAD_LENGTH - gpu_data->lane_pool.predicted_queue_length[lane_id]));
#else
		gpu_data->lane_pool.predicted_empty_space[lane_id] = min_device(
				gpu_data->lane_pool.last_time_empty_space[lane_id] + (gpu_data->lane_pool.speed[lane_id] * data_setting_gpu->ON_GPU_UNIT_TIME_STEPS),
				(data_setting_gpu->ON_GPU_ROAD_LENGTH - gpu_data->lane_pool.predicted_queue_length[lane_id]));
#endif
	}

	gpu_data->lane_pool.debug_data[lane_id] = gpu_data->lane_pool.predicted_empty_space[lane_id];
//update Tp

#ifdef ENABLE_CONSTANT
	gpu_data->lane_pool.accumulated_offset[lane_id] += gpu_data->lane_pool.speed[lane_id] * data_setting_gpu_constant.ON_GPU_UNIT_TIME_STEPS; //meter

	while (gpu_data->lane_pool.accumulated_offset[lane_id] >= gpu_data->lane_pool.lane_length[lane_id]) {
		gpu_data->lane_pool.accumulated_offset[lane_id] -= gpu_data->lane_pool.speed_history[gpu_data->lane_pool.Tp[lane_id]][lane_id] * data_setting_gpu_constant.ON_GPU_UNIT_TIME_STEPS;
		gpu_data->lane_pool.Tp[lane_id] += data_setting_gpu_constant.ON_GPU_UNIT_TIME_STEPS;
	}
#else

	gpu_data->lane_pool.accumulated_offset[lane_id] += gpu_data->lane_pool.speed[lane_id] * data_setting_gpu->ON_GPU_UNIT_TIME_STEPS; //meter

	while (gpu_data->lane_pool.accumulated_offset[lane_id] >= gpu_data->lane_pool.lane_length[lane_id]) {
		gpu_data->lane_pool.accumulated_offset[lane_id] -= gpu_data->lane_pool.speed_history[gpu_data->lane_pool.Tp[lane_id]][lane_id] * data_setting_gpu->ON_GPU_UNIT_TIME_STEPS;
		gpu_data->lane_pool.Tp[lane_id] += data_setting_gpu->ON_GPU_UNIT_TIME_STEPS;
	}
#endif

	//update queue length
	int queue_start = gpu_data->lane_pool.queue_length[lane_id] / data_setting_gpu->ON_GPU_VEHICLE_LENGTH;
	for (int queue_index = queue_start; queue_index < gpu_data->lane_pool.vehicle_counts[lane_id]; queue_index++) {
		if (gpu_data->lane_pool.vehicle_space[queue_index][lane_id]->entry_time <= gpu_data->lane_pool.Tp[lane_id]) {
			gpu_data->lane_pool.queue_length[lane_id] += data_setting_gpu->ON_GPU_VEHICLE_LENGTH;
		}
		else {
			break;
		}
	}
}

#endif

__device__ GPUVehicle* get_next_vehicle_at_node(GPUMemory* gpu_data, int node_id, int* lane_id, GPUSharedParameter* data_setting_gpu) {

	int maximum_waiting_time = -1;
//	int the_lane_id = -1;
	GPUVehicle* the_one_veh = NULL;

#ifdef ENABLE_CONSTANT
	for (int j = 0; j < data_setting_gpu_constant.ON_GPU_MAX_LANE_UPSTREAM; j++) {
#else
		for (int j = 0; j < data_setting_gpu->ON_GPU_MAX_LANE_UPSTREAM; j++) {
#endif
		int one_lane_id = gpu_data->node_pool.upstream[j][node_id];
		if (one_lane_id < 0) continue;

		/*
		 * Condition 1: The Lane is not NULL
		 * ----      2: Has Output Capacity
		 * ---       3: Is not blocked
		 * ---       4: Has vehicles
		 * ---       5: The vehicle can pass
		 */

		if (gpu_data->lane_pool.output_capacity[one_lane_id] > 0 && gpu_data->lane_pool.blocked[one_lane_id] == false && gpu_data->lane_pool.vehicle_counts[one_lane_id] > 0) {
			int time_diff = gpu_data->lane_pool.Tp[one_lane_id] - gpu_data->lane_pool.vehicle_space[0][one_lane_id]->entry_time;
			if (time_diff >= 0) {

				//if already the final move, then no need for checking next road
				if ((gpu_data->lane_pool.vehicle_space[0][one_lane_id]->next_path_index) >= (gpu_data->lane_pool.vehicle_space[0][one_lane_id]->whole_path_length)) {
					if (time_diff > maximum_waiting_time) {
						maximum_waiting_time = time_diff;
						*lane_id = one_lane_id;
						the_one_veh = gpu_data->lane_pool.vehicle_space[0][one_lane_id];
//						return gpu_data->lane_pool.vehicle_space[0][one_lane_id];
					}
				}
				else {
					int next_lane_index = gpu_data->lane_pool.vehicle_space[0][one_lane_id]->path_code[gpu_data->lane_pool.vehicle_space[0][one_lane_id]->next_path_index];
					int next_lane_id = gpu_data->node_pool.downstream[next_lane_index][node_id];

					/**
					 * Condition 6: The Next Lane has input capacity
					 * ---       7: The next lane has empty space
					 */
#ifdef ENABLE_CONSTANT
					if (gpu_data->lane_pool.input_capacity[next_lane_id] > 0 && gpu_data->lane_pool.predicted_empty_space[next_lane_id] > data_setting_gpu_constant.ON_GPU_VEHICLE_LENGTH) {
#else
						if (gpu_data->lane_pool.input_capacity[next_lane_id] > 0 && gpu_data->lane_pool.predicted_empty_space[next_lane_id] > data_setting_gpu->ON_GPU_VEHICLE_LENGTH) {
#endif
						if (time_diff > maximum_waiting_time) {
							maximum_waiting_time = time_diff;
							*lane_id = one_lane_id;
							the_one_veh = gpu_data->lane_pool.vehicle_space[0][one_lane_id];
//								return gpu_data->lane_pool.vehicle_space[0][one_lane_id];
						}
					}
					else {
						gpu_data->lane_pool.blocked[one_lane_id] = true;
					}
				}
			}
		}
	}

	return the_one_veh;
}

__global__ void supply_simulation_vehicle_passing(GPUMemory* gpu_data, int time_step, int node_length, GPUSharedParameter* data_setting_gpu) {
	int node_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (node_id >= node_length) return;

	for (int i = 0; i < gpu_data->node_pool.MAXIMUM_ACCUMULATED_FLOW[node_id]; i++) {
		int lane_id = -1;

		//Find A vehicle
		GPUVehicle* one_v = get_next_vehicle_at_node(gpu_data, node_id, &lane_id, data_setting_gpu);

		if (one_v == NULL || lane_id < 0) {
			//			printf("one_v == NULL\n");
			break;
		}

		if (one_v->entry_time <= gpu_data->lane_pool.Tp[lane_id]) {
			gpu_data->lane_pool.queue_length[lane_id] -= data_setting_gpu->ON_GPU_VEHICLE_LENGTH;
		}

		//Insert to next Lane
		if (gpu_data->lane_pool.vehicle_space[0][lane_id]->next_path_index >= gpu_data->lane_pool.vehicle_space[0][lane_id]->whole_path_length) {
			//the vehicle has finished the trip

			//			printf("vehicle %d finish trip at node %d,\n", one_v->vehicle_ID, node_id);
		}
		else {
			int next_lane_index = gpu_data->lane_pool.vehicle_space[0][lane_id]->path_code[gpu_data->lane_pool.vehicle_space[0][lane_id]->next_path_index];
			int next_lane_id = gpu_data->node_pool.downstream[next_lane_index][node_id];
			gpu_data->lane_pool.vehicle_space[0][lane_id]->next_path_index++;

			//it is very critical to update the entry time when passing
			gpu_data->lane_pool.vehicle_space[0][lane_id]->entry_time = time_step;

			//add the vehicle
			gpu_data->lane_pool.vehicle_passed_space[gpu_data->lane_pool.vehicle_passed_to_the_lane_counts[next_lane_id]][next_lane_id] = one_v;
			gpu_data->lane_pool.vehicle_passed_to_the_lane_counts[next_lane_id]++;

			gpu_data->lane_pool.input_capacity[next_lane_id]--;

#ifdef ENABLE_CONSTANT
			gpu_data->lane_pool.predicted_empty_space[next_lane_id] -= data_setting_gpu_constant.ON_GPU_VEHICLE_LENGTH;
#else
			gpu_data->lane_pool.predicted_empty_space[next_lane_id] -= data_setting_gpu->ON_GPU_VEHICLE_LENGTH;
#endif

			//			printf("time_step=%d,one_v->vehicle_ID=%d,lane_id=%d, next_lane_id=%d, next_lane_index=%d\n", time_step, one_v->vehicle_ID, lane_id, next_lane_id, next_lane_index);
		}

		//Remove from current Lane
		for (int j = 1; j < gpu_data->lane_pool.vehicle_counts[lane_id]; j++) {
			gpu_data->lane_pool.vehicle_space[j - 1][lane_id] = gpu_data->lane_pool.vehicle_space[j][lane_id];
		}

		gpu_data->lane_pool.vehicle_counts[lane_id]--;
		gpu_data->lane_pool.output_capacity[lane_id]--;
		gpu_data->lane_pool.flow[lane_id]++;
	}
}

__global__ void supply_simulation_after_vehicle_passing(GPUMemory* gpu_data, int time_step, int segment_length, GPUSharedParameter* data_setting_gpu) {
	int lane_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (lane_id >= segment_length) return;

//update queue length
//	bool continue_loop = true;
//	float queue_length = 0;
//	float acc_length_moving = gpu_data->lane_pool.accumulated_offset[lane_id];
//	int to_time_step = gpu_data->lane_pool.Tp[lane_id];
//
//	for (int i = 0; continue_loop && i < gpu_data->lane_pool.vehicle_counts[lane_id]; i++) {
//		if (gpu_data->lane_pool.vehicle_space[i][lane_id]->entry_time <= gpu_data->lane_pool.Tp[lane_id]) {
//			queue_length += data_setting_gpu->ON_GPU_VEHICLE_LENGTH;
//		}
//		else {
//			int entry_time = gpu_data->lane_pool.vehicle_space[i][lane_id]->entry_time;
//			for (int j = entry_time; i < to_time_step; i++) {
//				acc_length_moving -= gpu_data->lane_pool.speed_history[j][lane_id] * data_setting_gpu->ON_GPU_UNIT_TIME_STEPS;
//			}
//
//			if (acc_length_moving + queue_length >= gpu_data->lane_pool.lane_length[lane_id]) {
//				to_time_step = entry_time;
//				queue_length += data_setting_gpu->ON_GPU_VEHICLE_LENGTH;
//			}
//			else {
//				continue_loop = false;
//			}
//		}
//	}
//
////update queue length
//	gpu_data->lane_pool.queue_length[lane_id] = queue_length;

//update the queue history
	for (int i = 3; i > 0; i--) {
		gpu_data->lane_pool.his_queue_length[i][lane_id] = gpu_data->lane_pool.his_queue_length[i - 1][lane_id];
	}
	gpu_data->lane_pool.his_queue_length[0][lane_id] = gpu_data->lane_pool.queue_length[lane_id];

//update the empty space
//			if (gpu_data->lane_pool.new_vehicle_join_counts[lane_id] > 0) {
//				gpu_data->lane_pool.empty_space[lane_id] = std::min(gpu_data->lane_pool.speed[lane_id] * UNIT_TIME_STEPS, gpu_data->lane_pool.empty_space[lane_id])
//				if (gpu_data->lane_pool.empty_space[lane_id] < 0) gpu_data->lane_pool.empty_space[lane_id] = 0;
//			}
//			else {
	gpu_data->lane_pool.empty_space[lane_id] = gpu_data->lane_pool.empty_space[lane_id] + gpu_data->lane_pool.speed[lane_id] * data_setting_gpu->ON_GPU_UNIT_TIME_STEPS;
	gpu_data->lane_pool.empty_space[lane_id] = min_device(data_setting_gpu->ON_GPU_ROAD_LENGTH - gpu_data->lane_pool.queue_length[lane_id], gpu_data->lane_pool.empty_space[lane_id]);

}

__global__ void supply_simulated_results_to_buffer(GPUMemory* gpu_data, int time_step, int segment_length, SimulationResults* buffer, GPUSharedParameter* data_setting_gpu) {
	int buffer_index = time_step % data_setting_gpu->ON_GPU_GPU_TO_CPU_SIMULATION_RESULTS_COPY_BUFFER_SIZE;

	int lane_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (lane_id >= segment_length) return;

	buffer[buffer_index].flow[lane_id] = gpu_data->lane_pool.flow[lane_id];
	buffer[buffer_index].density[lane_id] = gpu_data->lane_pool.density[lane_id];
	buffer[buffer_index].speed[lane_id] = gpu_data->lane_pool.speed[lane_id];
	buffer[buffer_index].queue_length[lane_id] = gpu_data->lane_pool.queue_length[lane_id];
	buffer[buffer_index].counts[lane_id] = gpu_data->lane_pool.vehicle_counts[lane_id];

//	memcpy(buffer[buffer_index].flow, gpu_data->lane_pool.flow, sizeof(float) * data_setting_gpu->ON_GPU_LANE_SIZE);
//	memcpy(buffer[buffer_index].density, gpu_data->lane_pool.density, sizeof(float) * data_setting_gpu->ON_GPU_LANE_SIZE);
//	memcpy(buffer[buffer_index].speed, gpu_data->lane_pool.speed, sizeof(float) * data_setting_gpu->ON_GPU_LANE_SIZE);
//	memcpy(buffer[buffer_index].queue_length, gpu_data->lane_pool.queue_length, sizeof(float) * data_setting_gpu->ON_GPU_LANE_SIZE);
//	memcpy(buffer[buffer_index].counts, gpu_data->lane_pool.vehicle_counts, sizeof(int) * data_setting_gpu->ON_GPU_LANE_SIZE);

}

__device__ float min_device(float one_value, float the_other) {
	if (one_value < the_other) return one_value;
	else return the_other;
}

__device__ float max_device(float one_value, float the_other) {
	if (one_value > the_other) return one_value;
	else return the_other;
}
