///*
//* Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
//*
//* NVIDIA Corporation and its licensors retain all intellectual property and
//* proprietary rights in and to this software and related documentation and
//* any modifications thereto.  Any use, reproduction, disclosure, or distribution
//* of this software and related documentation without an express license
//* agreement from NVIDIA Corporation is strictly prohibited.
//*
//*
//*
//* This sample illustrates the usage of CUDA streams for overlapping
//* kernel execution with device/host memcopies.  The kernel is used to
//* initialize an array to a specific value, after which the array is
//* copied to the host (CPU) memory.  To increase performance, multiple
//* kernel/memcopy pairs are launched asynchronously, each pair in its
//* own stream.  Devices with Compute Capability 1.1 can overlap a kernel
//* and a memcopy as long as they are issued in different streams.  Kernels
//* are serialized.  Thus, if n pairs are launched, streamed approach
//* can reduce the memcopy cost to the (1/n)th of a single copy of the entire
//* data set.
//*
//* Additionally, this sample uses CUDA events to measure elapsed time for
//* CUDA calls.  Events are a part of CUDA API and provide a system independent
//* way to measure execution times on CUDA devices with approximately 0.5
//* microsecond precision.
//*
//* Elapsed times are averaged over nreps repetitions (10 by default).
//*
//*/
//
//const char *sSDKsample = "simpleStreams";
//
//#include <stdio.h>
//#include <cuda.h>
//#include <cuda_runtime_api.h>
//
//__global__ void init_array(int *g_data, int *factor, int num_iterations)
//{
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//
//	for(int i=0;i<num_iterations;i++)
//		g_data[idx] += *factor;	// non-coalesced on purpose, to burn time
//}
//
//int main(int argc, char *argv){
//
//	int nstreams = 4;               // number of streams for CUDA calls
//	int nreps = 10000;                 // number of times each experiment is repeated
//	int n = 16 * 1024 * 1024;       // number of ints in the data set
//	int nbytes = n * sizeof(int);   // number of data bytes
//	dim3 threads, blocks;           // kernel launch configuration
//	int niterations;	// number of iterations for the loop inside the kernel
//
//	int *d_a = 0, *d_c = 0;             // pointers to data and init value in the device memory
//	cudaMalloc((void**)&d_a, nbytes);
//	cudaMalloc((void**)&d_c, sizeof(int));
//
//	cudaStream_t stream1;
//	cudaStreamCreate(&stream1);
//
//	cudaEvent_t gpuBusy;
//	cudaEventCreate(&gpuBusy);
//	cudaError_t cudaFlag;
//
//	threads=dim3(512,1);
//	blocks=dim3(n/(nstreams*threads.x),1);
//	cudaMemset(d_a, 0, nbytes); // set device memory to all 0s, for testing correctness
//	int tHost = 0; int tDevice = 0;
//	for(int k = 0; k < nreps; k++)
//	{
//		cudaFlag = cudaEventQuery(gpuBusy);
//		if (cudaFlag == cudaSuccess) {
//			//Do some GPU stuffs.
//			init_array<<<blocks, threads, 0, stream1>>>(d_a, d_c, niterations);
//			cudaEventRecord(gpuBusy, stream1);
//			printf("\tgpu done: %d\n", tDevice);
//			tDevice++;
//		}
//
//		//Do some CPU staffs here;
//		printf("cpu done: %d\n", tHost);
//		tHost++;
//	}
//	system("PAUSE");
//}
