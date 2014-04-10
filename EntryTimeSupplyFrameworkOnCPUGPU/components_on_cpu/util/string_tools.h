/*
 * StringTools.h
 *
 *  Created on: Feb 6, 2014
 *      Author: xuyan
 */

#ifndef STRINGTOOLS_H_
#define STRINGTOOLS_H_

#include "shared_cpu_include.h"

class StringTools {
public:
	StringTools() {
	}
	~StringTools() {
	}

public:

	template<typename T>
	std::string toString(const T& value)
	{
	    std::ostringstream oss;
	    oss << value;
	    return oss.str();
	}
};


#endif /* STRINGTOOLS_H_ */
