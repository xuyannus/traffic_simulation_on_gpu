/*
 * TimeTools.h
 *
 *  Created on: Oct 30, 2013
 *      Author: xuyan
 */

#ifndef TIMETOOLS_H_
#define TIMETOOLS_H_

#include "shared_cpu_include.h"

#ifdef _WIN32

#include <time.h>

class TimeTools {

public:
	TimeTools() {
	}
	~TimeTools() {
	}

public:
	void start_profiling() {
		time(&start_time);
	}

	void end_profiling() {
		time(&end_time);
	}

	void output() {
		double cost = diff_ms(end_time, start_time);
		std::cout << "===================================" << std::endl;
		std::cout << "Simulation Time (seconds)" << cost << std::endl;
	}

	//unit is second
	double diff_ms(time_t t2, time_t t1) {
		return t2 - t1;
	}

private:
	time_t  start_time;
	time_t  end_time;
};
#else
#include <sys/time.h>

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
#endif

#endif /* TIMETOOLS_H_ */
