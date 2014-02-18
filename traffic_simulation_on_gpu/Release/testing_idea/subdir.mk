################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../testing_idea/memory_pointer.cpp 

CU_SRCS += \
../testing_idea/copy_memory_array_test.cu \
../testing_idea/main_old.cu \
../testing_idea/verySimpleDeviceHostSync.cu 

CU_DEPS += \
./testing_idea/copy_memory_array_test.d \
./testing_idea/main_old.d \
./testing_idea/verySimpleDeviceHostSync.d 

OBJS += \
./testing_idea/copy_memory_array_test.o \
./testing_idea/main_old.o \
./testing_idea/memory_pointer.o \
./testing_idea/verySimpleDeviceHostSync.o 

CPP_DEPS += \
./testing_idea/memory_pointer.d 


# Each subdirectory must supply rules for building sources it contributes
testing_idea/%.o: ../testing_idea/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-5.5/bin/nvcc -O3 -gencode arch=compute_30,code=sm_30 -odir "testing_idea" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-5.5/bin/nvcc --compile -O3 -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

testing_idea/%.o: ../testing_idea/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-5.5/bin/nvcc -O3 -gencode arch=compute_30,code=sm_30 -odir "testing_idea" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-5.5/bin/nvcc -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


