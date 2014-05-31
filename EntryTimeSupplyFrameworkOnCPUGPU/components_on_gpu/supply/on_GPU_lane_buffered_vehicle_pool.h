/*
 * on_GPU_lane_buffered_vehicle_pool.h
 *
 *  Created on: May 30, 2014
 *      Author: xuyan
 */

#ifndef ON_GPU_LANE_BUFFERED_VEHICLE_POOL_H_
#define ON_GPU_LANE_BUFFERED_VEHICLE_POOL_H_

#include "../../components_on_cpu/util/configurations_on_cpu.h"

class LaneBufferedVehiclePool {
public:

	//network
	int buffered_vehicle_space[kTotalBufferedVehicleSpace];
};


#endif /* ON_GPU_LANE_BUFFERED_VEHICLE_POOL_H_ */
