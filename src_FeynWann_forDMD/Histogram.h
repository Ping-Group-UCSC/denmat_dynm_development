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

#include <core/Util.h>
#include <core/matrix.h>
#include <vector>
#include <math.h>
#include <algorithm>

struct Histogram
{
	double Emin, dE, dEinv;
	int nE;
	std::vector<double> out;
	inline double Emax() const { return Emin + (nE-1)*dE; }
	
	Histogram(double Emin, double dE, double Emax);
	void addEvent(double E, double weight);
	
	//Alternate two-stage addEvent logic which is useful for arrays of similar histograms
	bool eventPrecalc(double E, int& iEvent, double& tEvent); //!< precalculate event location and return false if out of range
	inline void addEventPrecalc(int iEvent, double tEvent, double weight); //!< add event with precalculated location
	
	void allReduce(MPIUtil::ReduceOp op, bool safeMode=false); //collect over MPI
	void reduce(MPIUtil::ReduceOp op, int root=0); //collect over MPI
	void print(string fname, double Escale, double histScale) const; //write to file
};

struct Histogram2D
{
	double Emin, dE, dEinv, omegaMin, domega, domegaInv;
	int nE, nomega;
	std::vector<double> out; //nE by nomega with E inner dimension and omega outer
	inline double Emax() const { return Emin + (nE-1)*dE; }
	inline double omegaMax() const { return omegaMin + (nomega-1)*domega; }

	Histogram2D(double Emin, double dE, double Emax, double omegaMin, double domega, double omegaMax);
	void addEvent(double E, double omega, double weight);
	void allReduce(MPIUtil::ReduceOp op, bool safeMode=false); //collect over MPI
	void print(string fname, double Escale, double omegaScale, double histScale) const;
	
	Histogram2D(string fname, double Escale, double omegaScale, double histScale); //read back histogram written using print
	double interp1(double E, double omega) const; //return interpolated value
};

//----------------- Implementations --------------

inline void Histogram::addEventPrecalc(int iEvent, double tEvent, double weight)
{	double prefac = dEinv * weight;
	out[ iEvent ] += prefac * (1.-tEvent);
	out[iEvent+1] += prefac * tEvent;
}

#endif //FEYNWANN_HISTOGRAM_H
