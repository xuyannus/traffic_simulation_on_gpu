#!/bin/sh

echo "start testing the relationship between network order and performance"

for time in 1 2 3 4 5
do

network_file_path="data/exp2/network_100.dat"
network_flow_file_path="data/exp2/network_100_flow_rank.dat"
network_density_file_path="data/exp2/network_100_density_rank.dat"

demand_file_path="data/exp2/demand_100_100000.dat"
simulation_output_file_path="output/basic_ouput.txt"

./Release/ETS_CPU_AND_GPU_Source_Code_GPU 0 $network_file_path $demand_file_path $simulation_output_file_path > ./output/console_basic_network$time.txt
./Release/ETS_CPU_AND_GPU_Source_Code_GPU 0 $network_flow_file_path $demand_file_path $simulation_output_file_path > ./output/console_flow_order_network$time.txt
./Release/ETS_CPU_AND_GPU_Source_Code_GPU 0 $network_density_file_path $demand_file_path $simulation_output_file_path > ./output/console_density_order_network$time.txt

done

echo "done"

