################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../gpu_cpu/main_old.cu \
../gpu_cpu/verySimpleDeviceHostSync.cu 

CU_DEPS += \
./gpu_cpu/main_old.d \
./gpu_cpu/verySimpleDeviceHostSync.d 

OBJS += \
./gpu_cpu/main_old.o \
./gpu_cpu/verySimpleDeviceHostSync.o 


# Each subdirectory must supply rules for building sources it contributes
gpu_cpu/%.o: ../gpu_cpu/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-5.5/bin/nvcc -G -g -O0 -gencode arch=compute_30,code=sm_30 -odir "gpu_cpu" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-5.5/bin/nvcc --compile -G -O0 -g -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


