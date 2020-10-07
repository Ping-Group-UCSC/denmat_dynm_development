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

#ifndef FEYNWANN_HISTOGRAM_H
#define FEYNWANN_HISTOGRAM_H

#include <stdio.h>
#include <vector>
#include <math.h>
#include <algorithm>

struct Histogram
{
	double Emin, dE, dEinv;
	int nE;
	std::vector<double> out;
	inline double Emax() const { return Emin + (nE-1)*dE; }
	
	Histogram(double Emin, double dE, double Emax)
		: Emin(Emin), dE(dE), dEinv(1. / dE), nE(int(ceil((Emax - Emin) / dE))), out(nE, 0.)
	{}
	
	//Alternate two-stage addEvent logic which is useful for arrays of similar histograms
	bool eventPrecalc(double E, int& iEvent, double& tEvent)
	{
		double eCenter = (E - Emin)*dEinv;
		iEvent = floor(eCenter);
		if (iEvent<0 || iEvent + 1 >= nE) return false;
		tEvent = eCenter - iEvent;
		return true;
	}
	inline void addEventPrecalc(int iEvent, double tEvent, double weight)
	{
		double prefac = dEinv * weight;
		out[iEvent] += prefac * (1. - tEvent);
		out[iEvent + 1] += prefac * tEvent;
	}

	void addEvent(double E, double weight)
	{	//Linear splined histogram
		//--- E coordinate:
		int iEvent; double tEvent;
		if (not eventPrecalc(E, iEvent, tEvent)) return;
		//--- accumulate normalized linear spline:
		addEventPrecalc(iEvent, tEvent, weight);
	}

	void print(string fname, double Escale, double histScale) const
	{
		FILE* fp = fopen(fname.c_str(), "w");
		for (size_t i = 0; i < out.size(); i++)
			fprintf(fp, "%14.7le %14.7le\n", (Emin + i*dE)*Escale, out[i] * histScale);
		fclose(fp);
	}
};

#endif //FEYNWANN_HISTOGRAM_H
