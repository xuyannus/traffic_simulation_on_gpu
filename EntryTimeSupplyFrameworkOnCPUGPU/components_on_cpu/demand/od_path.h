/*
 * OD_Path.h
 *
 *  Created on: Jan 1, 2014
 *      Author: xuyan
 */

#ifndef OD_PATH_H_
#define OD_PATH_H_

#include "../util/shared_cpu_include.h"
#include "../util/configurations_on_cpu.h"

class ODPairPATH {

public:
	int path_id;
	int od_id;
	std::vector<int> link_ids;
	std::bitset<kMaxLaneCodingLength> route_code;
public:
	static bool load_in_all_OD_Paths(std::vector<ODPairPATH*>& all_od_pair_paths, const std::string od_path_file_path);
};

bool ODPairPATH::load_in_all_OD_Paths(std::vector<ODPairPATH*>& all_od_pair_paths, const std::string od_path_file_path) {
	//very important, ID starts from 0
	int od_path_id = 0;

	std::string line;
	std::ifstream myfile(od_path_file_path.c_str());

	if (myfile.is_open()) {
		while (getline(myfile, line)) {
			if (line.empty() || line.compare(0, 1, "#") == 0) {
				continue;
			}

			std::vector<std::string> temp_elems;
			network_reading_split(line, ':', temp_elems);
			assert(temp_elems.size() == 4);

			ODPairPATH* one_path = new ODPairPATH();
			one_path->path_id = od_path_id;
			od_path_id++;
			one_path->od_id = atoi(temp_elems[1].c_str());

			std::vector<std::string> temp_elems2;
			network_reading_split(temp_elems[2], ',', temp_elems2);
			for (int ii = 0; ii < temp_elems2.size(); ii++) {
				int link_id = atoi(temp_elems2[ii].c_str());
				one_path->link_ids.push_back(link_id);
			}

			std::string line = temp_elems[3];
			for (int ii = 0; ii < line.length(); ii++) {
				if (line.at(ii) == '0') {
					one_path->route_code.set(ii, 0);
				} else {
					one_path->route_code.set(ii, 1);
				}
			}

			all_od_pair_paths.push_back(one_path);
		}
		myfile.close();
	} else {
		std::cout << "Unable to open OD Path file:" << od_path_file_path << std::endl;
		return false;
	}

	std::cout << "How many OD pair routes are loaded?:" << all_od_pair_paths.size() << std::endl;
	return true;
}

#endif /* OD_PATH_H_ */
