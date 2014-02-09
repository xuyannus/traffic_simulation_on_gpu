#!/bin/sh

for demand_level in 10000 50000
do 

network_file_path="data/exp1_network/network_10_rank.dat"
demand_file_path="data/exp1/demand_10_$demand_level.dat"
simulation_output_file_path="output/simulated_outputs_mode0_$demand_level.txt"

echo "do serial code"
./Debug/ETS_CPU_AND_GPU_Source_Code 1 $network_file_path $demand_file_path $simulation_output_file_path > ./output/console_$demand_level.txt
echo "serial code done"


echo "do partition code"
for EXECUTION_MODE in 4
do

for Partition_MODE in 2 4 6 8 10 20 30 40 60 121
do

network_file_path="data/exp1_network/network_10_rank.dat_$Partition_MODE"
simulation_output_file_path="output/simulated_outputs_mode$EXECUTION_MODE.$demand_level.$Partition_MODE.txt"

./Debug/ETS_CPU_AND_GPU_Source_Code $EXECUTION_MODE $network_file_path $demand_file_path $simulation_output_file_path > ./output/console_$Partition_MODE.$EXECUTION_MODE.$demand_level.txt

java -jar ./RMSNToolkit.jar output/simulated_outputs_mode0_$demand_level.txt $simulation_output_file_path > ./output/RMSN_$Partition_MODE.$EXECUTION_MODE.$demand_level.txt

done

done
echo "partition code done"

echo "calculate RMSN"

done


