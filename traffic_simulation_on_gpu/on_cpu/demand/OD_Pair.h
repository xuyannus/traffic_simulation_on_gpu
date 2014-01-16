/*
 * OD_Pair.h
 *
 *  Created on: Jan 1, 2014
 *      Author: xuyan
 */

#ifndef OD_PAIR_H_
#define OD_PAIR_H_

#include "../util/shared_cpu_include.h"

using namespace std;

class OD_Pair_PATH;

class OD_Pair {

public:
	int od_pair_id;
	int from_node_id;
	int to_node_id;

	vector<OD_Pair_PATH*> all_paths;

public:
	static bool load_in_all_ODs(vector<OD_Pair*>& all_od_pairs, string od_file_path);

};

bool OD_Pair::load_in_all_ODs(vector<OD_Pair*>& all_od_pairs, string od_file_path) {
	
	string line;
	ifstream myfile(od_file_path.c_str());

	//very important, ID starts from 0
	int od_id = 0;

	if (myfile.is_open()) {
		while (getline(myfile, line)) {
			if (line.empty() || line.compare(0, 1, "#")==0) {
				continue;
			}

			std::vector<std::string> elems;
			network_reading_split(line, ',', elems);
			assert(elems.size() == 2);

			OD_Pair* one = new OD_Pair();
			one->od_pair_id = od_id;
			od_id++;

			one->from_node_id = atoi(elems[0].c_str());
			one->to_node_id = atoi(elems[1].c_str());

			all_od_pairs.push_back(one);
		}
		myfile.close();
	}
	else {
		cout << "Unable to open OD Pair file:" << od_file_path << endl;
	}

	cout << "OD Pair Loaded" << endl;
	cout << "ODs:" << all_od_pairs.size() << endl;
	cout << "-------------------------------------" << endl;
	
	return true;
	
}

#endif /* OD_PAIR_H_ */
