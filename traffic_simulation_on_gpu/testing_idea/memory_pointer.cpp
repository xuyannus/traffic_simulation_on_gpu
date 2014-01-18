//#include <stdio.h>
//#include <iostream>
//#include <fstream>
//#include <stdlib.h>
//#include <stdio.h>
//
//class Vehicle {
//public:
//	int x;
//	int y;
//	int z;
//};
//
//int main(int argc, char *argv) {
//
//	int VEHICLE_SIZE = 10;
//
//	Vehicle* list_v = (Vehicle*) malloc(sizeof(Vehicle) * VEHICLE_SIZE);
//
//	for (int i = 0; i < VEHICLE_SIZE; i++) {
//		Vehicle* one_v = (Vehicle*)(list_v + i * sizeof(Vehicle));
//		one_v->x = i;
//		one_v->y = i;
//		one_v->z = i;
//	}
//
//	for (int i = 0; i < VEHICLE_SIZE; i++) {
//		Vehicle* one_v = (Vehicle*)(list_v + i * sizeof(Vehicle));
//		std::cout << "one_v->x:" << one_v->x << std::endl;
//		std::cout << "one_v->y:" << one_v->y << std::endl;
//		std::cout << "one_v->z:" << one_v->z << std::endl;
//	}
//
//	return 0;
//}
