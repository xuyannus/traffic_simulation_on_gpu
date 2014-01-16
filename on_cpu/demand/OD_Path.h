/*
 * OD_Path.h
 *
 *  Created on: Jan 1, 2014
 *      Author: xuyan
 */

#ifndef OD_PATH_H_
#define OD_PATH_H_

#include "../util/shared_cpu_include.h"
#include "../../main/shared_data.h"

using namespace std;

class OD_Pair_PATH {

public:
	int path_id;
	int od_id;

	vector<int> link_ids;
	std::bitset<100> route_code;

public:
	static bool load_in_all_OD_Paths(vector<OD_Pair_PATH*>& all_od_pair_paths, string od_path_file_path);

};

bool OD_Pair_PATH::load_in_all_OD_Paths(vector<OD_Pair_PATH*>& all_od_pair_paths, string od_path_file_path) {
	
	string line;
	ifstream myfile(od_path_file_path.c_str());

	//very important, ID starts from 0
	int od_path_id = 0;

//	int count=0;
	if (myfile.is_open()) {
		while (getline(myfile, line)) {
			//printf("%d ", count++);
			if (line.empty() || line.compare(0, 1, "#")==0) {
				continue;
			}

			std::vector<std::string> elems;
			network_reading_split(line, ':', elems);
			assert(elems.size() == 4);

			OD_Pair_PATH* one_path = new OD_Pair_PATH();
			one_path->path_id = od_path_id;
			od_path_id++;

			one_path->od_id = atoi(elems[1].c_str());
			std::vector<std::string> elems2;
			network_reading_split(elems[2], ',', elems2);

			for (int ii = 0; ii < elems2.size(); ii++) {
				int link_id = atoi(elems2[ii].c_str());
				one_path->link_ids.push_back(link_id);
			}

			std::string line = elems[3];
			for (int ii = 0; ii < line.length(); ii++) {
				if (line.at(ii) == '0') {
					one_path->route_code.set(ii, 0);
				}
				else {
					one_path->route_code.set(ii, 1);
				}
			}

			all_od_pair_paths.push_back(one_path);
		}
		myfile.close();
	}
	else {
		cout << "Unable to open OD Path file:" << od_path_file_path << endl;
	}

	cout << "OD Pair Routes Loaded" << endl;
	cout << "Routes:" << all_od_pair_paths.size() << endl;
	cout << "-------------------------------------" << endl;
	
	return true;
}

#endif /* OD_PATH_H_ */
