/*
 * Network.h
 *
 *  Created on: Jan 1, 2014
 *      Author: xuyan
 */

#ifndef NETWORK_H_
#define NETWORK_H_

#include "../util/shared_cpu_include.h"

#include "Link.h"
#include "Node.h"

using namespace std;

class Network {

public:
	std::vector<Node*> all_nodes;
	std::vector<Link*> all_links;

	std::map<string, bool> road_connect_broken;
	std::map<int, Node*> node_mapping;

public:
	static bool load_network(Network* network, string network_file_path);

};

std::vector<std::string> &network_reading_split(const std::string &s, char delim, std::vector<std::string> &elems) {
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		elems.push_back(item);
	}
	return elems;
}

bool Network::load_network(Network* network, string network_file_path) {

	//clear the data firstly
	network->all_nodes.clear();
	network->all_links.clear();
	network->road_connect_broken.clear();
	network->node_mapping.clear();

	string line;
	ifstream myfile(network_file_path.c_str());
	if (myfile.is_open()) {
		while (getline(myfile, line)) {

			if (line.empty() || line.compare(0, 1, "#") == 0) {
				continue;
			}

			if (line.compare(0, 5, "NODE:") == 0) {
				std::vector<std::string> elems;
				network_reading_split(line, ':', elems);

				assert(elems.size() == 4);

				Node* one_node = new Node();
				one_node->node_id = atoi(elems[1].c_str());
				one_node->x = atoi(elems[2].c_str());
				one_node->y = atoi(elems[3].c_str());

				network->all_nodes.push_back(one_node);
				network->node_mapping[one_node->node_id] = one_node;
			}

			else if (line.compare(0, 5, "LINK:") == 0) {
				std::vector<std::string> elems;
				network_reading_split(line, ':', elems);

				std::cout << "a line: " << line << std::endl;

				assert(elems.size() == 4);

				Link* one_link = new Link();
				one_link->link_id = atoi(elems[1].c_str());
				one_link->from_node = network->node_mapping[atoi(elems[2].c_str())];
				one_link->to_node = network->node_mapping[atoi(elems[3].c_str())];

				assert(one_link->from_node != NULL);
				assert(one_link->to_node != NULL);

				one_link->from_node->downstream_links.push_back(one_link);
				one_link->to_node->upstream_links.push_back(one_link);

				network->all_links.push_back(one_link);
			}

			else if (line.compare(0, 6, "LINK_C") == 0) {
//				std::vector<std::string> elems;
//				network_reading_split(line, ':', elems);
//
//				assert(elems.size() == 4);
//
//				string key = elems[1].append(",");
//				key = key.append(elems[2]);

//				bool value = (atoi(elems[3].c_str()) == 1) ? true : false;
//
//				network->road_connect_broken[key] = value;
			}
		}
		myfile.close();
	}
	else {
		cout << "Unable to open network file:" << network_file_path << endl;
	}

	cout << "Network Loaded" << endl;
	cout << "Nodes:" << network->all_nodes.size() << endl;
	cout << "Links:" << network->all_links.size() << endl;
	cout << "road_connect_broken size:" << network->road_connect_broken.size() << endl;
	cout << "-------------------------------------" << endl;

	return true;
}

#endif /* NETWORK_H_ */
