/*
 * TimeTools.h
 *
 *  Created on: Oct 30, 2013
 *      Author: xuyan
 */

#ifndef TIMETOOLS_H_
#define TIMETOOLS_H_

#include "shared_cpu_include.h"
#include <sys/time.h>
//typedef struct timeval {
//  long tv_sec;
//  long tv_usec;
//} timeval;

class TimeTools {
public:
	TimeTools() {
	}
	~TimeTools() {
	}

public:
	void start_profiling() {
		gettimeofday(&start_time, NULL);
	}

	void end_profiling() {
		gettimeofday(&end_time, NULL);
	}

	void output() {
		double cost = diff_ms(end_time, start_time);
		std::cout << "===================================" << std::endl;
		std::cout << "Simulation Time (ms10E-3)" << cost << std::endl;
	}

	double diff_ms(timeval t1, timeval t2) {
		return (((t1.tv_sec - t2.tv_sec) * 1000.0) + (t1.tv_usec - t2.tv_usec) / 1000.0);
	}

private:
	timeval start_time;
	timeval end_time;
};

#endif /* TIMETOOLS_H_ */
