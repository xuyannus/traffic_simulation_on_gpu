/*
 * shared_cpu_include.h
 *
 *  Created on: Jan 1, 2014
 *      Author: xuyan
 */

#ifndef SHARED_GPU_DATA_H_
#define SHARED_GPU_DATA_H_

//for small network
//const int kLaneSize = 220;
//const int kNodeSize = 121;

//for large network
const int kLaneSize = 20200;
const int kNodeSize = 10201;

const int kStartTimeSteps = 0;
const int kEndTimeSteps = 1000;

const int kUnitTimeStep = 1; //sec
const int kTotalTimeSteps = (kEndTimeSteps - kStartTimeSteps) / kUnitTimeStep;

const int kQueueLengthHistory = 4;

const int kMaxLaneDownstream = 2;
const int kMaxLaneUpstream = 2;

//Length Related
const int kRoadLength = 1000; //meter
const int kVehicleLength = 5; //meter
const int kMaxVehiclePerLane = kRoadLength / kVehicleLength;
const int kVehicleMaxLoadingOneTime = 2;

//Speed Related
const float kAlpha = 1;
const float kBeta = 0.5;
const float kMaxDensity = 1.0; //vehicle on road
const float kMinDensity = 10.0 * kVehicleLength / kRoadLength; //vehicle on road

const int kMaxSpeed = 30;
const int kMinSpeed = 1;

const int kLaneInputCapacityPerTimeStep = 1;
const int kLaneOutputCapacityPerTimeStep = 1;

const int kMaxRouteLength = 100;
const int kMaxLaneCodingLength = 500;
const int kGPUToCPUSimulationResultsCopyBufferSize = 100;

#endif /* SHARED_GPU_DATA_H_ */
