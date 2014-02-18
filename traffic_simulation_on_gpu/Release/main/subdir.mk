################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../main/Experiment1_on_cpu.cpp \
../main/main_on_cpu.cpp 

CU_SRCS += \
../main/main.cu 

CU_DEPS += \
./main/main.d 

OBJS += \
./main/Experiment1_on_cpu.o \
./main/main.o \
./main/main_on_cpu.o 

CPP_DEPS += \
./main/Experiment1_on_cpu.d \
./main/main_on_cpu.d 


# Each subdirectory must supply rules for building sources it contributes
main/%.o: ../main/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-5.5/bin/nvcc -O3 -gencode arch=compute_30,code=sm_30 -odir "main" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-5.5/bin/nvcc -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

main/%.o: ../main/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-5.5/bin/nvcc -O3 -gencode arch=compute_30,code=sm_30 -odir "main" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-5.5/bin/nvcc --compile -O3 -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


