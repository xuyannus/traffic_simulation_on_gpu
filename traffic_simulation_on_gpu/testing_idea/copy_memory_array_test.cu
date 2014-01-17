#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <fstream>

const int VEHICLE_LENGTH_ON_ROAD = 100;
const int ROADS_SIZE = 10;

class Vehicle {
public:
	int x;
	int y;
	int z;
};

class VehicleGroup {
public:
	Vehicle all_vehicles_on_lane[VEHICLE_LENGTH_ON_ROAD];
};

class GPU_Memory {
public:
	void* open_space;

	VehicleGroup* lanes_groups[ROADS_SIZE];
	int simulated_result;
};

__global__ void simulate_on_GPU(GPU_Memory* data_on_gpu, void* open_space_gpu) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx == 1) {

		for (int i = 0; i < ROADS_SIZE; i++) {
			data_on_gpu->lanes_groups[i] = (VehicleGroup*)(data_on_gpu->open_space + i * sizeof(VehicleGroup));
		}

		data_on_gpu->simulated_result = data_on_gpu->lanes_groups[0]->all_vehicles_on_lane[4].x;
	}
}

GPU_Memory* data_on_gpu;
void* open_space_gpu;

void init_gpu();

int main(int argc, char *argv) {

	init_gpu();

	return 0;
}

using namespace std;

void init_gpu() {
	data_on_gpu = NULL;

	cout << "start:" << std::endl;

	GPU_Memory* data_on_cpu = new GPU_Memory();
	for (int i = 0; i < ROADS_SIZE; i++) {
		data_on_cpu->lanes_groups[i] = new VehicleGroup();

		for (int j = 0; j < VEHICLE_LENGTH_ON_ROAD; j++) {
			data_on_cpu->lanes_groups[i]->all_vehicles_on_lane[j].x = j;
			data_on_cpu->lanes_groups[i]->all_vehicles_on_lane[j].y = j;
			data_on_cpu->lanes_groups[i]->all_vehicles_on_lane[j].z = j;
		}
	}

	data_on_cpu->open_space = (void *) malloc(sizeof(VehicleGroup) * ROADS_SIZE);

	for (int i = 0; i < ROADS_SIZE; i++) {
		memcpy(data_on_cpu->open_space + i * sizeof(VehicleGroup), data_on_cpu->lanes_groups[i], sizeof(VehicleGroup));
	}

	cout << "CPU side done" << std::endl;

	data_on_cpu->simulated_result = 2;

	if (cudaMalloc(&data_on_gpu, sizeof(GPU_Memory)) != cudaSuccess) {
		cerr << "cudaMalloc(&gpu_data, sizeof(GPUMemory)) failed" << endl;
	}

	if (cudaMalloc(&(open_space_gpu), sizeof(VehicleGroup) * ROADS_SIZE) != cudaSuccess) {
		cerr << "cudaMalloc(&gpu_data, sizeof(GPUMemory)) failed" << endl;
	}

	cout << "GPU side cudaMalloc done" << std::endl;

	cudaMemcpy(data_on_gpu, data_on_cpu, sizeof(GPU_Memory), cudaMemcpyHostToDevice);
	cudaMemcpy(data_on_gpu->open_space, data_on_cpu->open_space, sizeof(VehicleGroup) * ROADS_SIZE, cudaMemcpyHostToDevice);

	cout << "start 3:" << std::endl;

	simulate_on_GPU<<<1, 1>>>(data_on_gpu, open_space_gpu);

	cout << "start 4:" << std::endl;

	int simualted_data = 0;

	cudaMemcpy(&simualted_data, &data_on_gpu->simulated_result, sizeof(int), cudaMemcpyDeviceToHost);

	cout << "simualted_data:" << simualted_data << std::endl;
}
