/*
 * Network.h
 *
 *  Created on: Jan 1, 2014
 *      Author: xuyan
 */

#ifndef NETWORK_H_
#define NETWORK_H_

#include "../util/shared_cpu_include.h"

#include "link.h"
#include "node.h"
#include "link_connection.h"

class Network {
public:
	int node_size;
	Node** all_nodes;

	int link_size;
	Link** all_links;

	int link_conn_size;
	LinkConnection** all_link_conn;

	std::map<int, Node*> node_mapping;
	std::map<int, Link*> link_mapping;

public:
	static bool load_network(Network& network, const std::string network_file_path);
};

std::vector<std::string> &network_reading_split(const std::string &s, char delim, std::vector<std::string> &elems) {
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		elems.push_back(item);
	}
	return elems;
}

bool Network::load_network(Network& network, const std::string network_file_path) {

	//clear the data firstly
	network.node_mapping.clear();
	network.link_mapping.clear();

	std::string line;
	std::ifstream myfile(network_file_path.c_str());

	if (myfile.is_open()) {
		while (getline(myfile, line)) {
			if (line.empty() || line.compare(0, 1, "#") == 0) {
				continue;
			}

			if (line.compare(0, 11, "PARAM_NODE:") == 0) {
				std::vector<std::string> temp_elems;
				network_reading_split(line, ':', temp_elems);

				assert(temp_elems.size() == 2);

				network.node_size = atoi(temp_elems[1].c_str());
				network.all_nodes = new Node*[network.node_size];

			} else if (line.compare(0, 5, "NODE:") == 0) {
				std::vector<std::string> temp_elems;
				network_reading_split(line, ':', temp_elems);

//				std::cout << "temp_elems.size():" << temp_elems.size() << std::endl;
				assert(temp_elems.size() == 8);

				Node* one_node = new Node();
				one_node->node_id = atoi(temp_elems[1].c_str());
				one_node->up_link_start_index = atoi(temp_elems[2].c_str());
				one_node->up_link_end_index = atoi(temp_elems[3].c_str());
				one_node->up_lane_start_index = atoi(temp_elems[6].c_str());
				one_node->up_lane_end_index = atoi(temp_elems[7].c_str());

				network.all_nodes[one_node->node_id] = one_node;
				network.node_mapping[one_node->node_id] = one_node;

			} else if (line.compare(0, 11, "PARAM_LANE:") == 0) {
				std::vector<std::string> temp_elems;
				network_reading_split(line, ':', temp_elems);

				assert(temp_elems.size() == 2);

				network.link_size = atoi(temp_elems[1].c_str());
				network.all_links = new Link*[network.link_size];
			}

			else if (line.compare(0, 5, "LANE:") == 0) {
				std::vector<std::string> temp_elems;
				network_reading_split(line, ':', temp_elems);

				assert(temp_elems.size() == 9);

				Link* one_link = new Link();

				one_link->link_id = atoi(temp_elems[1].c_str());

				one_link->link_connection_start = atoi(temp_elems[2].c_str());
				one_link->link_connection_end = atoi(temp_elems[3].c_str());

				one_link->vehicle_start = atoi(temp_elems[4].c_str());
				one_link->vehicle_end = atoi(temp_elems[5].c_str());

				one_link->buffered_vehicle_start = atoi(temp_elems[6].c_str());
				one_link->buffered_vehicle_end = atoi(temp_elems[7].c_str());

				one_link->length = atof(temp_elems[8].c_str());

				network.all_links[one_link->link_id] = one_link;
				network.link_mapping[one_link->link_id] = one_link;
			}

			else if (line.compare(0, 20, "PARAM_LANECONNECTION") == 0) {
				std::vector<std::string> temp_elems;
				network_reading_split(line, ':', temp_elems);

				assert(temp_elems.size() == 2);

				network.link_conn_size = atoi(temp_elems[1].c_str());
				network.all_link_conn = new LinkConnection*[network.link_conn_size];
			}

			else if (line.compare(0, 14, "LANECONNECTION") == 0) {
				static int lin_conn_ID = 0;

				std::vector<std::string> temp_elems;
				network_reading_split(line, ':', temp_elems);

				assert(temp_elems.size() == 3);

				LinkConnection* one_link_conn = new LinkConnection();
				one_link_conn->from_link_id = atoi(temp_elems[1].c_str());
				one_link_conn->to_link_id = atoi(temp_elems[2].c_str());

				network.all_link_conn[lin_conn_ID] = one_link_conn;
				lin_conn_ID++;
			}
		}
		myfile.close();
	} else {
		std::cout << "Unable to open network file:" << network_file_path << std::endl;
	}

	std::cout << "-------------------------------------" << std::endl;
	std::cout << "Network Loaded" << std::endl;
	std::cout << "Nodes:" << network.node_size << std::endl;
	std::cout << "Links:" << network.link_size << std::endl;
	std::cout << "Link Connections:" << network.link_conn_size << std::endl;
	std::cout << "-------------------------------------" << std::endl;

	return true;
}

#endif /* NETWORK_H_ */
