/*-------------------------------------------------------------------
Copyright 2018 Ravishankar Sundararaman

This file is part of JDFTx.

JDFTx is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

JDFTx is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with JDFTx.  If not, see <http://www.gnu.org/licenses/>.
-------------------------------------------------------------------*/

#include <core/Util.h>
#include <fstream>
#include <cmath>
#include "InputMap.h"

extern void environmentSubstitute(string& s); //defined in jdftx/commands/parser.cpp

InputMap::InputMap(string filename)
{	std::ifstream ifs(filename.c_str());
	if(!ifs.is_open())
		die("Could not open system file '%s' for reading.\n", filename.c_str());
	while(!ifs.eof())
	{	string line; getline(ifs, line); //line-by-line processing (comments can now be inline)
		trim(line);
		istringstream iss(line);
		string name; string val;
		if(iss >> name >> val)
		{	//Substitute enviornment variables:
			if(mpiWorld->isHead())
				environmentSubstitute(val);
			mpiWorld->bcast(val);
			(*this)[name] = val;
		}
	}
	ifs.close();
}

bool InputMap::has(string key) const
{	return (find(key) != end());
}

double InputMap::get(string key, double defaultVal) const
{	auto iter = find(key);
	if(iter == end()) //not found
	{	if(std::isnan(defaultVal)) //no default provided
		{	die("\nCould not find required entry '%s' in input.\n", key.c_str());
		}
		else return defaultVal;
	}
	return atof(iter->second.c_str());
}

vector3<> InputMap::getVector(string key, vector3<> defaultVal) const
{	auto iter = find(key);
	if(iter == end()) //not found
	{	if(std::isnan(defaultVal[0])) //no default provided
		{	die("\nCould not find required entry '%s' in input.\n", key.c_str());
		}
		else return defaultVal;
	}
	//Parse value string with comma as a delimiter:
	vector3<> result;
	istringstream iss(iter->second);
	for(int k=0; k<3; k++)
	{	string token;
		getline(iss, token, ',');
		result[k] = atof(token.c_str());
	}
	return result;
}

bool InputMap::getBool(string key, bool defaultVal) const
{	auto iter = find(key);
	if(iter == end()) //not found
		return defaultVal;
	else
	{	string input = iter->second;
		if((input != "yes") and (input != "no"))
			die("\n%s must be 'yes' or 'no'\n", key.c_str());
		return (input == "yes");
	}
}

string InputMap::getString(string key, string defaultVal) const
{	auto iter = find(key);
	if (iter == end()) return defaultVal; //not found
	//if (iter == end()) die("\nCould not find required entry '%s' in input.\n", key.c_str())
	return iter->second;
}
