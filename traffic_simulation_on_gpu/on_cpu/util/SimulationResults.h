/*
 * SimulationResults.h
 *
 *  Created on: Jan 3, 2014
 *      Author: xuyan
 */

#ifndef SIMULATIONRESULTS_H_
#define SIMULATIONRESULTS_H_

#include "../../main/shared_data.h"
//#include "../util/shared_cpu_include.h"
//
class SimulationResults {
public:

	float flow[LANE_SIZE];
	float density[LANE_SIZE];
	float speed[LANE_SIZE];
	float queue_length[LANE_SIZE];
	int counts[LANE_SIZE];

};

#endif /* SIMULATIONRESULTS_H_ */
