/*
 * SimulationResults.h
 *
 *  Created on: Jan 3, 2014
 *      Author: xuyan
 */

#ifndef SIMULATIONRESULTS_H_
#define SIMULATIONRESULTS_H_

#include "../util/configurations_on_cpu.h"

class SimulationResults {
public:

	float flow[kLaneSize];
	float density[kLaneSize];
	float speed[kLaneSize];
	float queue_length[kLaneSize];
	int counts[kLaneSize];

};

#endif /* SIMULATIONRESULTS_H_ */
