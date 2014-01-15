/*
 * Vehicle.h
 *
 *  Created on: Jan 1, 2014
 *      Author: xuyan
 */

#ifndef CPUVEHICLE_H_
#define CPUVEHICLE_H_

#include "../util/shared_cpu_include.h"

using namespace std;

class Vehicle {

public:
	int vehicle_id;
	int od_id;
	int path_id;
	int entry_time;

public:
	static bool load_in_all_vehicles(vector<Vehicle*>& all_vehicles, string demand_file_path);

};

bool Vehicle::load_in_all_vehicles(vector<Vehicle*>& all_vehicles, string demand_file_path) {

	string line;
	ifstream myfile(demand_file_path.c_str());

	//very important, ID starts from 0

	if (myfile.is_open()) {
		while (getline(myfile, line)) {
			if (line.empty() || line.compare(0, 1, "#")==0) {
				continue;
			}

			std::vector<std::string> elems;
			network_reading_split(line, ':', elems);
			assert(elems.size() == 4);

			Vehicle* one_vec = new Vehicle();
			one_vec->vehicle_id = atoi(elems[0].c_str());
			one_vec->od_id = atoi(elems[1].c_str());
			one_vec->path_id = atoi(elems[2].c_str());
			one_vec->entry_time = atoi(elems[3].c_str());

			all_vehicles.push_back(one_vec);
		}
		myfile.close();
	}
	else {
		cout << "Unable to open OD Path file:" << demand_file_path << endl;
	}

	cout << "Vehicle Loaded" << endl;
	cout << "Vehicles:" << all_vehicles.size() << endl;
	cout << "-------------------------------------" << endl;

	return true;
}

#endif /* CPUVEHICLE_H_ */
