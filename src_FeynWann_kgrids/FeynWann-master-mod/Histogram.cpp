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

#include "Histogram.h"
#include <core/Util.h>
#include <core/matrix.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>

//------------- class Histogram --------------

Histogram::Histogram(double Emin, double dE, double Emax)
: Emin(Emin), dE(dE), dEinv(1./dE), nE(int(ceil((Emax-Emin)/dE))), out(nE, 0.)
{
}

void Histogram::addEvent(double E, double weight)
{	//Linear splined histogram
	//--- E coordinate:
	int iEvent; double tEvent;
	if(not eventPrecalc(E, iEvent, tEvent)) return;
	//--- accumulate normalized linear spline:
	addEventPrecalc(iEvent, tEvent, weight);
}

bool Histogram::eventPrecalc(double E, int& iEvent, double& tEvent)
{	double eCenter = (E-Emin)*dEinv;
	iEvent = floor(eCenter);
	if(iEvent<0 || iEvent+1>=nE) return false;
	tEvent = eCenter - iEvent;
	return true;
}

void Histogram::allReduce(MPIUtil::ReduceOp op, bool safeMode)
{	if(mpiWorld->nProcesses()>1)
		mpiWorld->allReduceData(out, op, safeMode);
}

void Histogram::reduce(MPIUtil::ReduceOp op, int root)
{	if(mpiWorld->nProcesses()>1)
		mpiWorld->reduceData(out, op, root);
}

void Histogram::print(string fname, double Escale, double histScale) const
{	if(!mpiWorld->isHead()) return;
	ofstream ofs(fname.c_str());
	for(size_t i=0; i<out.size(); i++)
		ofs << (Emin+i*dE)*Escale << "\t" << out[i]*histScale << '\n';
}

//------------- class Histogram2D --------------

Histogram2D::Histogram2D(double Emin, double dE, double Emax, double omegaMin, double domega, double omegaMax)
: Emin(Emin), dE(dE), dEinv(1./dE), omegaMin(omegaMin), domega(domega), domegaInv(1./domega),
nE(int(ceil((Emax-Emin)/dE))), nomega(int(ceil((omegaMax-omegaMin)/domega))), out(nE*nomega, 0.)
{
}

void Histogram2D::addEvent(double E, double omega, double weight)
{	//Linear splined 2D Histogram
	//--- E coordinate:
	double eCenter = (E-Emin)*dEinv;
	int ie = floor(eCenter);
	if(ie<0 || ie+1>=nE) return;
	double te = eCenter - ie;
	//--- omega coordinate:
	double oCenter = (omega-omegaMin)*domegaInv;
	int io = floor(oCenter);
	if(io<0 || io+1>=nomega) return;
	double to = oCenter - io;
	//--- accumulate normalized linear spline:
	double prefac = dEinv * domegaInv * weight;
	double eContrib0 = prefac * (1.-te);
	double eContrib1 = prefac * te;
	out[( io )*nE+( ie )] += eContrib0 * (1.-to);
	out[( io )*nE+(ie+1)] += eContrib1 * (1.-to);
	out[(io+1)*nE+( ie )] += eContrib0 * to;
	out[(io+1)*nE+(ie+1)] += eContrib1 * to;
}

void Histogram2D::allReduce(MPIUtil::ReduceOp op, bool safeMode)
{	if(mpiWorld->nProcesses()>1)
		mpiWorld->allReduceData(out, op, safeMode);
}

void Histogram2D::print(string fname, double Escale, double omegaScale, double histScale) const
{	if(!mpiWorld->isHead()) return;
	ofstream ofs(fname.c_str());
	//Print in octave/matlab mat format for ease:
	//--- E grid:
	ofs << "# name: E\n";
	ofs << "# type: matrix\n";
	ofs << "# rows: " << nE << "\n";
	ofs << "# columns: 1\n";
	for(int e=0; e<nE; e++)
		ofs << (Emin+e*dE)*Escale << "\n";
	ofs <<"\n";
	//--- omega grid:
	ofs << "# name: omega\n";
	ofs << "# type: matrix\n";
	ofs << "# rows: " << nomega << "\n";
	ofs << "# columns: 1\n";
	for(int o=0; o<nomega; o++)
		ofs << (omegaMin+o*domega)*omegaScale << "\n";
	ofs <<"\n";
	//--- Histogram data:
	ofs << "# name: histData\n";
	ofs << "# type: matrix\n";
	ofs << "# rows: " << nE << "\n";
	ofs << "# columns: " << nomega << "\n";
	for(int e=0; e<nE; e++)
	{	for(int o=0; o<nomega; o++)
		{	if(o) ofs << ' ';
			ofs << out[o*nE+e] * histScale;
		}
		ofs <<"\n";
	}
}

Histogram2D::Histogram2D(string fname, double Escale, double omegaScale, double histScale)
{	logPrintf("Reading '%s':", fname.c_str()); logFlush();
	ifstream ifs(fname.c_str());
	string line, buf;
	//--- E grid:
	getline(ifs, line); assert(line == "# name: E");
	getline(ifs, line); assert(line == "# type: matrix");
	getline(ifs, line); { istringstream iss(line); iss >> buf >> buf >> nE; logPrintf(" nE = %d ", nE); }
	getline(ifs, line); assert(line == "# columns: 1");
	std::vector<double> Egrid(nE);
	for(int iE=0; iE<nE; iE++)
	{	getline(ifs, line);
		istringstream iss(line);
		double Ein; iss >> Ein;
		Egrid[iE] = Ein/Escale;
	}
	Emin = Egrid.front();
	dE = (Egrid.back()-Emin)/(nE-1);
	dEinv = 1./dE;
	logPrintf(" dE = %le ", dE);
	getline(ifs, line); assert(line == "");
	//--- omega grid:
	getline(ifs, line); assert(line == "# name: omega");
	getline(ifs, line); assert(line == "# type: matrix");
	getline(ifs, line); { istringstream iss(line); iss >> buf >> buf >> nomega; logPrintf(" nomega = %d ", nomega); }
	getline(ifs, line); assert(line == "# columns: 1");
	std::vector<double> omegaGrid(nomega);
	for(int iomega=0; iomega<nomega; iomega++)
	{	getline(ifs, line);
		istringstream iss(line);
		double omegaIn; iss >> omegaIn;
		omegaGrid[iomega] = omegaIn/omegaScale;
	}
	omegaMin = omegaGrid.front();
	domega = (omegaGrid.back()-omegaMin)/(nomega-1);
	domegaInv = 1./domega;
	logPrintf(" domega = %le ", domega);
	getline(ifs, line); assert(line == "");
	//--- Histogram data:
	getline(ifs, line); assert(line == "# name: histData");
	getline(ifs, line); assert(line == "# type: matrix");
	getline(ifs, line); { istringstream iss(line); int nRows; iss >> buf >> buf >> nRows; assert(nRows == nE); }
	getline(ifs, line); { istringstream iss(line); int nCols; iss >> buf >> buf >> nCols; assert(nCols == nomega); }
	out.resize(nE*nomega);
	for(int e=0; e<nE; e++)
	{	getline(ifs, line);
		istringstream iss(line);
		for(int o=0; o<nomega; o++)
		{	double histIn; iss >> histIn;
			out[o*nE+e] = histIn/histScale;
		}
	}
	logPrintf("\n");
}

double Histogram2D::interp1(double E, double omega) const
{	//--- E coordinate:
	double eCenter = (E-Emin)*dEinv;
	int ie = floor(eCenter);
	if(ie<0 || ie+1>=nE) return 0.;
	double te = eCenter - ie;
	//--- omega coordinate:
	double oCenter = (omega-omegaMin)*domegaInv;
	int io = floor(oCenter);
	if(io<0 || io+1>=nomega) return 0.;
	double to = oCenter - io;
	//--- interpolate linearly:
	return
		  out[( io )*nE+( ie )] * (1.-to) * (1.-te)
		+ out[( io )*nE+(ie+1)] * (1.-to) * te
		+ out[(io+1)*nE+( ie )] * to * (1.-te)
		+ out[(io+1)*nE+(ie+1)] * to * te;
}

