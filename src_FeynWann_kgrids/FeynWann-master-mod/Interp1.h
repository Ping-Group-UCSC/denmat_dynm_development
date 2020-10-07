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

#ifndef FEYNWANN_INTERP1_H
#define FEYNWANN_INTERP1_H

#include <core/Util.h>
#include <cmath>

struct Interp1
{
	std::vector<string> headerVals; //header values for each columns (read from file, if available, but not used by this class)
	std::vector<double> xGrid; //common x values (uniform grid)
	std::vector<std::vector<double> > yGrid; //y values per column
	double xMin, dx, dxInv; //for speeding up interpolation
	
	//Read from file which has a single line header, interpolate along columns
	//xScale and yScale allow for unit conversions in the input data
	void init(string fname, double xScale, double yScale);
	
	inline double operator()(int iColumn, double x) const
	{	assert(iColumn<int(yGrid.size()));
		const std::vector<double>& yCol = yGrid[iColumn];
		double fx = dxInv * (x - xMin);
		if(fx <= 0.) return yCol.front();
		if(fx >= yCol.size()-1) return yCol.back();
		int ix = floor(fx); double tx = fx - ix; //find integer and fractional coordinate
		return (1.-tx)*yCol[ix] + tx*yCol[ix+1];
	}
	
	//Single y-column version
	inline double operator()(double x) const
	{	assert(yGrid.size()==1);
		return (*this)(0, x);
	}
};

#endif //FEYNWANN_INTERP1_H
