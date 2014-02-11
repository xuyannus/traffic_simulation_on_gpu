/*
 * shared_cpu_include.h
 *
 *  Created on: Jan 1, 2014
 *      Author: xuyan
 */

#ifndef SHARED_GPU_DATA_H_
#define SHARED_GPU_DATA_H_

//for small network
//const int LANE_SIZE = 220;
//const int NODE_SIZE = 121;

//for large network
const int LANE_SIZE = 20200;
const int NODE_SIZE = 10201;

const int START_TIME_STEPS = 0;
const int END_TIME_STEPS = 1000;

//for exp 1
//const int END_TIME_STEPS = 3600;
const int UNIT_TIME_STEPS = 1; //sec
const int TOTAL_TIME_STEPS = (END_TIME_STEPS - START_TIME_STEPS) / UNIT_TIME_STEPS;

//const int TOTAL_VEHICLES = 5000;

const int QUEUE_LENGTH_HISTORY = 4;
//const int MAX_LANE_CONNECTION = 2;

const int MAX_LANE_DOWNSTREAM = 2;
const int MAX_LANE_UPSTREAM = 2;

//Length Related
const int ROAD_LENGTH = 1000; //meter
const int VEHICLE_LENGTH = 5; //meter
const int MAX_VEHICLE_PER_LANE = 200;
const int VEHICLE_MAX_LOADING_ONE_TIME = 2;

//Speed Related
const float Alpha = 1;
const float Beta = 0.5;
const float Max_Density = 1.0; //vehicle on road
const float Min_Density = 10.0 * VEHICLE_LENGTH / ROAD_LENGTH; //vehicle on road
const int MAX_SPEED = 30;
const int MIN_SPEED = 1;

const int LANE_INPUT_CAPACITY_TIME_STEP = 1;
const int LANE_OUTPUT_CAPACITY_TIME_STEP = 1;

const int MAX_ROUTE_LENGTH = 100;

const int MAXIMUM_LANE_CODING_LENGTH = 500;

const int GPU_TO_CPU_SIMULATION_RESULTS_COPY_BUFFER_SIZE = 1;

#endif /* SHARED_GPU_DATA_H_ */
