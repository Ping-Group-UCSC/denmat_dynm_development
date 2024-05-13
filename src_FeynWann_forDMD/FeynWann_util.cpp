/*-------------------------------------------------------------------
Copyright 2022 Ravishankar Sundararaman

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

#include "FeynWann.h"
#include "FeynWann_internal.h"
#include <core/LatticeUtils.h>
#include <core/Units.h>
#include <core/Random.h>


vector3<> FeynWann::randomVector(MPIUtil* mpiUtil)
{	vector3<> v;
	for(int iDir=0; iDir<3; iDir++)
		v[iDir] = Random::uniform();
	if(mpiUtil) mpiUtil->bcast(&v[0], 3);
	return v;
}


void FeynWann::symmetrize(matrix3<>& m) const
{	matrix3<> mOut;
	matrix3<> invR = inv(R);
	int nSym = 0;
	for(const SpaceGroupOp& op: sym)
	{	matrix3<> rot = R * op.rot * invR; //convert to Cartesian
		//Exclude rotations that don't leave fields invariant
		if(fwp.Bext.length_squared())
		{	if((fwp.Bext - rot*fwp.Bext).length() > symmThreshold)
				continue;
		}
		if(fwp.EzExt)
		{	vector3<> Eext(0., 0., fwp.EzExt);
			if((Eext - rot*Eext).length() > symmThreshold)
				continue;
		}
		mOut += rot * m * (~rot);
		nSym++;
	}
	m = mOut * (1./nSym);
	//Set near-zero to exact zero:
	double mCut = 1e-14*sqrt(trace((~m)*m));
	for(int i=0; i<3; i++)
		for(int j=0; j<3; j++)
			if(fabs(m(i,j)) < mCut)
				m(i,j) = 0.;
}


//Elementwise std::pow of a matrix
template<typename PowType> matrix3<> powElemWise(const matrix3<>& m, PowType n)
{	matrix3<> result;
	for(int i=0; i<3; i++)
		for(int j=0; j<3; j++)
			result(i,j) = std::pow(m(i,j), n);
	return result;
}

//Elementwise std::pow of a vector
template<typename PowType> vector3<> powElemWise(const vector3<>& m, PowType n)
{	vector3<> result;
	for(int i=0; i<3; i++)
		result[i] = std::pow(m[i], n);
	return result;
}

//Report a tensor with error estimates
void reportResult(const std::vector<matrix3<>>& result, string resultName, double unit, string unitName, FILE* fp, bool invAvg)
{	matrix3<> sum, sumSq; int N = 0;
	for(size_t block=0; block<result.size(); block++)
	{	N++;
		matrix3<> term = invAvg ? inv(result[block]) : result[block];
		sum += term;
		sumSq += powElemWise(term, 2);
	}
	matrix3<> resultMean = (1./N)*sum;
	matrix3<> resultStd = powElemWise((1./N)*sumSq - powElemWise(resultMean,2), 0.5); //element-wise std. deviation
	if(invAvg)
	{	resultMean = inv(resultMean); //harmonic matrix mean (inverse of mean matrix inverse)
		resultStd = resultMean * resultStd * resultMean; //propagate error in reciprocal
	}
	//Print result:
	for(int i=0; i<3; i++)
	{	char mOpen[] = "/|\\", mClose[] = "\\|/";
		fprintf(fp, "%20s%c", i==1 ? (resultName + " = ").c_str() : "", mOpen[i]);
		for(int j=0; j<3; j++) fprintf(fp, " %12lg", resultMean(i,j)/unit);
		if(N>1)
		{	fprintf(fp, " %c%5s%c", mClose[i], i==1 ? " +/- " : "", mOpen[i]);
			for(int j=0; j<3; j++) fprintf(fp, " %12lg", fabs(resultStd(i,j))/unit);
			fprintf(fp, " %c %s\n", mClose[i], i==1 ? unitName.c_str() : "");
		}
		else fprintf(fp, " %c %s\n", mClose[i], i==1 ? unitName.c_str() : "");
	}
	fprintf(fp, "\n");
}

//Report a vector with error estimates
void reportResult(const std::vector<vector3<>>& result, string resultName, double unit, string unitName, FILE* fp, bool invAvg)
{	vector3<> sum, sumSq; int N = 0;
	for(size_t block=0; block<result.size(); block++)
	{	N++;
		vector3<> term = invAvg ? powElemWise(result[block], -1) : result[block];
		sum += term;
		sumSq += powElemWise(term, 2);
	}
	vector3<> resultMean = (1./N)*sum;
	vector3<> resultStd = powElemWise((1./N)*sumSq - powElemWise(resultMean,2), 0.5); //element-wise std. deviation
	if(invAvg)
	{	resultMean = powElemWise(resultMean, -1); //harmonic elementwise vector mean
		resultStd = Diag(powElemWise(resultMean, 2)) * resultStd; //propagate error in reciprocal
	}
	//Print result:
	fprintf(fp, "%17s = [", resultName.c_str());
	for(int i=0; i<3; i++) fprintf(fp, " %12lg", resultMean[i]/unit);
	if(N>1)
	{	fprintf(fp, " ] +/- [");
		for(int i=0; i<3; i++) fprintf(fp, " %12lg", fabs(resultStd[i])/unit);
	}
	fprintf(fp, " ] %s\n", unitName.c_str());
}

//Report a scalar with error estimates:
void reportResult(const std::vector<double>& result, string resultName, double unit, string unitName, FILE* fp, bool invAvg)
{	double sum = 0., sumSq = 0.; int N = 0;
	for(size_t block=0; block<result.size(); block++)
	{	N++;
		double term = invAvg ? 1./result[block] : result[block];
		sum += term;
		sumSq += term*term;
	}
	double resultMean = sum/N;
	double resultStd = sqrt(sumSq/N - std::pow(resultMean,2));
	if(invAvg)
	{	resultMean = 1./resultMean; //harmonic mean
		resultStd *= std::pow(resultMean,2); //propagate error in reciprocal
	}
	if(N>1)
		fprintf(fp, "%17s = %16lg +/- %16lg %s\n", resultName.c_str(), resultMean/unit, fabs(resultStd)/unit, unitName.c_str());
	else
		fprintf(fp, "%17s = %12lg %s\n", resultName.c_str(), resultMean/unit, unitName.c_str());
}



std::vector<vector3<int>> readCellMap(string fname)
{	logPrintf("Reading '%s' ... ", fname.c_str()); logFlush();
	ifstream ifs(fname); if(!ifs.is_open()) die("could not open file.\n");
	string headerLine; getline(ifs, headerLine); //read and ignore header line
	std::vector<vector3<int>> cellMap;
	vector3<int> cm; //lattice coords version (store)
	vector3<> Rcm; //cartesian version (ignore)
	while(ifs >> cm[0] >> cm[1] >> cm[2] >> Rcm[0] >> Rcm[1] >> Rcm[2])
		cellMap.push_back(cm);
	ifs.close();
	logPrintf("done.\n"); logFlush();
	return cellMap;
}


std::vector<matrix> readCellWeights(string fname, int nCells, int nAtoms, int nBands)
{	logPrintf("Reading '%s' ... ", fname.c_str()); logFlush();
	matrix cellWeightsAll(nAtoms*nBands, nCells);
	cellWeightsAll.read_real(fname.c_str());
	//--- split to matrix per cell:
	std::vector<matrix> cellWeights(nCells);
	for(int iCell=0; iCell<nCells; iCell++)
	{	cellWeights[iCell] = cellWeightsAll(0,nAtoms*nBands, iCell,iCell+1);
		cellWeights[iCell].reshape(nAtoms,nBands);
	}
	logPrintf("done.\n");
	return cellWeights;
}


diagMatrix readPhononBasis(string fname)
{	logPrintf("Reading '%s' ... ", fname.c_str()); logFlush();
	ifstream ifs(fname); if(!ifs.is_open()) die("could not open file.\n");
	string headerLine; getline(ifs, headerLine); //read and ignore header line
	diagMatrix invsqrtM;
	while(!ifs.eof())
	{	string line; getline(ifs, line);
		trim(line);
		if(!line.length()) continue;
		istringstream iss(line);
		string spName; int atom; vector3<> disp; double M;
		iss >> spName >> atom >> disp[0] >> disp[1] >> disp[2] >> M;
		if(!iss.fail())
		{	invsqrtM.push_back(1./sqrt(M*amu));
		}
	}
	logPrintf("done.\n"); logFlush();
	return invsqrtM;		
}

//JX
std::vector<vector3<>> readPhononBasisFromFile(string fname){
	logPrintf("Reading '%s' ... ", fname.c_str()); logFlush();
	ifstream ifs(fname); if (!ifs.is_open()) die("could not open file.\n");
	string headerLine; getline(ifs, headerLine); //read and ignore header line
	std::vector<vector3<>> phononBasis;
	while (!ifs.eof()){
		string line; getline(ifs, line);
		trim(line);
		if (!line.length()) continue;
		istringstream iss(line);
		string spName; int atom; vector3<> disp; double M;
		iss >> spName >> atom >> disp[0] >> disp[1] >> disp[2] >> M;
		if (!iss.fail()){
			phononBasis.push_back(disp);
		}
	}
	logPrintf("done.\n"); logFlush();
	return phononBasis;
}
//JX
