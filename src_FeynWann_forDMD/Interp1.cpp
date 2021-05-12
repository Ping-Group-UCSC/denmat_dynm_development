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

#include "Interp1.h"

void Interp1::init(string fname, double xScale, double yScale)
{	logPrintf("Reading '%s': ", fname.c_str()); logFlush();
	ifstream ifs(fname.c_str());
	if(!ifs.is_open()) die("Could not open or find '%s' for reading.\n", fname.c_str());
	string line; //read line by line
	//Read header:
	getline(ifs, line);
	trim(line);
	if(line[0]=='#') //Have header (multi column mode)
	{	istringstream iss(line);
		string comment; iss >> comment; //ignore
		while(!iss.eof())
		{	string headerVal; iss >> headerVal;
			if(iss.fail()) break;
			headerVals.push_back(headerVal);
		}
		line.clear(); //use up line
	}
	else //No header (sigle column mode)
		headerVals.push_back(string()); //don't use up line already read in
	logPrintf("%lu columns, ", headerVals.size()); logFlush();
	//Read data:
	yGrid.resize(headerVals.size());
	while(!ifs.eof())
	{	if(!line.length()) getline(ifs, line); //read line (if no unused line already available)
		if(!line.length()) break; //quit on empty line, otherwise must have full data
		istringstream iss(line);
		//Read x:
		double x; iss >> x;
		if(iss.fail()) die("Error reading x[%lu]\n", xGrid.size())
		xGrid.push_back(x * xScale);
		//read y values:
		for(size_t iy=0; iy<headerVals.size(); iy++)
		{	double y; iss >> y;
			if(iss.fail()) die("Error reading y[%lu][%lu]\n", iy, xGrid.size())
			yGrid[iy].push_back(y * yScale);
		}
		line.clear(); //use up line
	}
	//Make sure x is an uniform grid:
	xMin = xGrid[0];
	dx = (xGrid.back() - xMin) / (xGrid.size() - 1);
	dxInv = 1./dx;
	for(size_t i=0; i<xGrid.size(); i++)
		if(fabs(dxInv*(xGrid[i]-xMin) - i) > 1e-2)
			die("x is not a uniform grid\n")
	logPrintf("%lu rows.\n", xGrid.size()); logFlush();
}
