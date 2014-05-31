/*
 * OD_Pair.h
 *
 *  Created on: Jan 1, 2014
 *      Author: xuyan
 */

#ifndef OD_PAIR_H_
#define OD_PAIR_H_

#include "../util/shared_cpu_include.h"
#include "../network/network.h"

//forward declaration
class ODPairPATH;

class ODPair {
public:
	int od_pair_id;
	int from_node_id;
	int to_node_id;

	std::vector<ODPairPATH*> all_paths;
public:
	static bool load_in_all_ODs(std::vector<ODPair*>& all_od_pairs, const std::string od_file_path);
};

bool ODPair::load_in_all_ODs(std::vector<ODPair*>& all_od_pairs, const std::string od_file_path) {
	//very important, ID starts from 0
	int od_id = 0;

	std::string line;
	std::ifstream myfile(od_file_path.c_str());

	if (myfile.is_open()) {
		while (getline(myfile, line)) {
			if (line.empty() || line.compare(0, 1, "#")==0) {
				continue;
			}

			std::vector<std::string> temp_elems;
			network_reading_split(line, ':', temp_elems);

//			std::cout << "line:" << line << std::endl;
			assert(temp_elems.size() == 2);

			ODPair* one = new ODPair();
			one->od_pair_id = od_id;
			od_id ++;

			one->from_node_id = atoi(temp_elems[0].c_str());
			one->to_node_id = atoi(temp_elems[1].c_str());

			all_od_pairs.push_back(one);
		}
		myfile.close();
	}
	else {
		std::cout << "Unable to open OD Pair file:" << od_file_path << std::endl;
		return false;
	}

	std::cout << "How many OD pairs are loaded?:" << all_od_pairs.size() << std::endl;
	return true;
}

#endif /* OD_PAIR_H_ */
