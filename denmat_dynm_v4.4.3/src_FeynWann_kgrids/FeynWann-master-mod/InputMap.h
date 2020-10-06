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

#ifndef FEYNWANN_INPUTMAP_H
#define FEYNWANN_INPUTMAP_H

#include <core/string.h>
#include <core/vector3.h>
#include <map>

//Parse simple input file into a dictionary
class InputMap : std::map<string,string>
{
public:
	InputMap(string filename);
	double get(string key, double defaultVal=NAN) const;
	vector3<> getVector(string key, vector3<> defaultVal=vector3<>(NAN)) const; //!< comma-delimited vector
	string getString(string key) const;
};

#endif //FEYNWANN_INPUTMAP_H