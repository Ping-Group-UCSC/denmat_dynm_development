/*-------------------------------------------------------------------
Copyright 2019 Adela Habib, Ravishankar Sundararaman

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
#include <core/matrix.h>
#include "FeynWann.h"
#include "InputMap.h"
#include <core/Units.h>

//Read a list of k-points from a file
std::vector<vector3<>> readKpointsFile(string fname)
{	std::vector<vector3<>> kArr;
	logPrintf("Reading '%s' ... ", fname.c_str()); logFlush();
	ifstream ifs(fname); if(!ifs.is_open()) die("could not open file.\n");
	while(!ifs.eof())
	{	string line; getline(ifs, line);
		trim(line);
		if(!line.length()) continue;
		//Parse line
		istringstream iss(line);
		string key; iss >> key;
		if(key == "kpoint")
		{	vector3<> k;
			iss >> k[0] >> k[1] >> k[2];
			kArr.push_back(k);
		}
	}
	ifs.close();
	logPrintf("done.\n");
	return kArr;
}

//Write debug code within process() to examine arbitrary e-ph properties along a k- or q-path
struct DebugEph
{	int bandStart, bandStop; //optional band range read in from input
	int modeStart, modeStop; //optional mode range read in from input
	double Mtot;
	
	//Previously computed quantities using single-k version to test against transformed ones:
	FeynWann::StateE e1, e2;
	FeynWann::StatePh ph;
	FeynWann::MatrixEph m;
	bool spinAvailable;
	double degeneracyThreshold;
	
	DebugEph(int bandStart, int bandStop, int modeStart, int modeStop, const FeynWannParams& fwp)
	: bandStart(bandStart), bandStop(bandStop), modeStart(modeStart), modeStop(modeStop),
	spinAvailable(fwp.needSpin), degeneracyThreshold(fwp.degeneracyThreshold)
	{
	}
	
	void process(const FeynWann::MatrixEph& mEph)
	{	const diagMatrix& E1 = mEph.e1->E;
		const diagMatrix& E2 = mEph.e2->E;
		const diagMatrix& omegaPh = mEph.ph->omega;
		
		//---- Single k compute debug ----
				
		/*
		//---- Phonon frequency debug ----
		logPrintf("OMEGAPH(%lf,%lf,%lf):", mEph.ph->q[0], mEph.ph->q[1], mEph.ph->q[2]);
		for(const double omega: omegaPh)
			logPrintf(" %11.8lf", omega);
		logPrintf("\n");
		logFlush();
		*/

		//---- overlap matrix element debug ----
		logPrintf("|ovlp_{}(k1,k2)|^2: ");
		matrix ovlp = dagger(mEph.e1->U) * mEph.e2->U;
		for (int b1 = bandStart; b1 < bandStop; b1++){
			for (int b2 = bandStart; b2 < bandStop; b2++){
				double m2 = 0, ndeg = 0;
				for (int d1 = bandStart; d1 < bandStop; d1++)
				for (int d2 = bandStart; d2 < bandStop; d2++)
				if ((fabs(E1[d1] - E1[b1]) < 1e-6) and(fabs(E2[d2] - E2[b2]) < 1e-6)){
					m2 += ovlp(d1, d2).norm();
					ndeg += 1;
				}
				logPrintf("%lg ", m2 / ndeg);
			}
		}
		logPrintf("\n"); logFlush();
		
		//---- e-ph matrix element debug ----
		logPrintf("|g|(%lf,%lf,%lf):", mEph.ph->q[0], mEph.ph->q[1], mEph.ph->q[2]);
		
		for(int iMode=modeStart; iMode<modeStop; iMode++){
			double gNormCur = 0.0;
			gNormCur = (mEph.M[iMode](bandStart,bandStart)).norm();
			/*
			for(int b1=bandStart; b1<bandStop; b1++)
			{	for(int b2=bandStart; b2<bandStop; b2++)
				{	//if ( (fabs(E1[b1] - E1[bandStart]) < 1e-5) and (fabs(E2[b2] - E2[bandStart]) < 1e-5) )
					//gNormCur += (sqrt(2.*Mtot*omegaPh[iMode])*mEph.M[iMode](b1,b2)).norm();	
					gNormCur += (mEph.M[iMode](b1,b2)).norm();	
				}
			}*/
			logPrintf(" %11.8lf", sqrt(gNormCur)); 
		}
		logPrintf("\n");
		logFlush();
		
		//---- Spin commutator debug ---
		if(spinAvailable)
		{	matrix S1z = degenerateProject(mEph.e1->S[2], E1);
			matrix S2z = degenerateProject(mEph.e2->S[2], E2);
			double Gsq = 0., SGsq = 0.; //traced over specified band and mode range
			const matrix& G = mEph.M[modeStart];
			matrix SGcomm = S1z * G - G * S2z;
			for(int mode=modeStart; mode<modeStop; mode++)
			{	const matrix& G = mEph.M[mode];
				matrix SGcomm = S1z * G - G * S2z;
				for(int b1=bandStart; b1<bandStop; b1++)
				for(int b2=bandStart; b2<bandStop; b2++)
				{	Gsq += G(b1,b2).norm();
					SGsq += SGcomm(b1,b2).norm();
				}
			}
			logPrintf("SGcommDEBUG: %le %le\n", sqrt(Gsq), sqrt(SGsq));
		}
	}
	static void ePhProcess(const FeynWann::MatrixEph& mEph, void* params)
	{	((DebugEph*)params)->process(mEph);
	}
	
	inline matrix degenerateProject(const matrix& M, const diagMatrix& E)
	{	matrix out = M;
		complex* outData = out.data();
		for(int b2=0; b2<out.nCols(); b2++)
			for(int b1=0; b1<out.nRows(); b1++)
			{	if(fabs(E[b1] - E[b2]) > degeneracyThreshold) (*outData) = 0;
				outData++;
			}
		return out;
	}
	
	//Sum over degenerate subspace of electrons at one k
	inline matrix degSqSum(const matrix& M, const diagMatrix& E)
	{	matrix ret(M.nRows(), M.nCols());
		//Loop over left degenerate subspace:
		for(int b1start=0; b1start<E.nRows();)
		{	int b1stop=b1start+1;
			while(b1stop<E.nRows() and E[b1stop]<E[b1start]+degeneracyThreshold)
				b1stop++;
			//Loop over right degenerate subspace:
			for(int b2start=0; b2start<E.nRows();)
			{	int b2stop=b2start+1;
				while(b2stop<E.nRows() and E[b2stop]<E[b2start]+degeneracyThreshold)
					b2stop++;
				//Compute sum over subspace:
				double out = 0.;
				for(int b1=b1start; b1<b1stop; b1++)
					for(int b2=b2start; b2<b2stop; b2++)
						out += M(b1,b2).norm();
				//Set sum over subspace:
				for(int b1=b1start; b1<b1stop; b1++)
					for(int b2=b2start; b2<b2stop; b2++)
						ret.set(b1,b2, out);
				b2start = b2stop;
			}
			b1start = b1stop;
		}
		return ret;
	}
	
	//Sum over degenerate subspace of electrons and phonons at k1,k2
	inline std::vector<matrix> degSqSum(const std::vector<matrix>& M, const diagMatrix& E1, const diagMatrix& E2, const diagMatrix& omegaPh)
	{	std::vector<matrix> ret(M.size(), matrix(M[0].nRows(), M[0].nCols()));
		//Loop over omegaPh subspace:
		for(int modeStart=0; modeStart<omegaPh.nRows();)
		{	int modeStop=modeStart+1;
			while(modeStop<omegaPh.nRows() and omegaPh[modeStop]<omegaPh[modeStart]+degeneracyThreshold)
				modeStop++;
			//Loop over E1 degenerate subspace:
			for(int b1start=0; b1start<E1.nRows();)
			{	int b1stop=b1start+1;
				while(b1stop<E1.nRows() and E1[b1stop]<E1[b1start]+degeneracyThreshold)
					b1stop++;
				//Loop over E2 degenerate subspace:
				for(int b2start=0; b2start<E2.nRows();)
				{	int b2stop=b2start+1;
					while(b2stop<E2.nRows() and E2[b2stop]<E2[b2start]+degeneracyThreshold)
						b2stop++;
					//Compute sum over subspace:
					double out = 0.;
					for(int mode=modeStart; mode<modeStop; mode++)
						for(int b1=b1start; b1<b1stop; b1++)
							for(int b2=b2start; b2<b2stop; b2++)
								out += M[mode](b1,b2).norm();
					//Set sum over subspace:
					for(int mode=modeStart; mode<modeStop; mode++)
						for(int b1=b1start; b1<b1stop; b1++)
							for(int b2=b2start; b2<b2stop; b2++)
								ret[mode].set(b1,b2, out);
					b2start = b2stop;
				}
				b1start = b1stop;
			}
			modeStart = modeStop;
		}
		return ret;
	}
};

int main(int argc, char** argv)
{	
	InitParams ip = FeynWann::initialize(argc, argv, "Print electron phonon matrix element, |g_q|.");
	
	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(ip.inputFilename);
	const vector3<> k1 = inputMap.getVector("k1");
	string k2file = inputMap.getString("k2file"); //file containing list of k2 points (like a bandstruct.kpoints file)
	int bandStart = inputMap.get("bandStart", 0);
	int bandStop = inputMap.get("bandStop", 0); //replaced with nBands below if 0
	int modeStart = inputMap.get("modeStart", 0);
	int modeStop = inputMap.get("modeStop", 0); //replaced with nModes below if 0
	FeynWannParams fwp(&inputMap);
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("k1 = "); k1.print(globalLog, " %lg ");
	logPrintf("k2file = %s\n", k2file.c_str());
	logPrintf("bandStart = %d\n", bandStart);
	logPrintf("bandStop = %d\n", bandStop);
	logPrintf("modeStart = %d\n", modeStart);
	logPrintf("modeStop = %d\n", modeStop);
	fwp.printParams();
	
	//Read k-points:
	std::vector<vector3<>> k2arr = readKpointsFile(k2file);
	logPrintf("Read %lu k-points from '%s'\n", k2arr.size(), k2file.c_str());
	
	//Initialize FeynWann:
	fwp.needPhonons = true;
	fwp.ePhHeadOnly = true; //so as to debug k-path alone
	fwp.needSpin = true;
	FeynWann fw(fwp);
	if(!bandStop) bandStop = fw.nBands;
	if(!modeStop) modeStop = fw.nModes;
	
	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		fw.free();
		FeynWann::finalize();
		return 0;
	}
	logPrintf("\n");
	diagMatrix invsqrtM = fw.invsqrtM;
	double Mtot = 0.0;
	for(int iMode=0; iMode< fw.nModes; iMode++)
	{	Mtot += 1./std::pow(invsqrtM[iMode],2);
	}
	Mtot *= 1./3;
	DebugEph src(bandStart, bandStop, modeStart, modeStop, fwp);
	src.Mtot = Mtot;
	fw.eCalc(k1, src.e1);
	if(mpiGroup->isHead())
	{	logPrintf("E1[eV]: ");
		(src.e1.E*(1./eV)).print(globalLog);
	}
	for(vector3<> k2: k2arr)
	{	//Compute single k quantities:
		fw.eCalc(k2, src.e2);
		fw.phCalc(k1-k2, src.ph);
		fw.ePhCalc(src.e1, src.e2, src.ph, src.m);
		
		if(mpiGroup->isHead()) src.process(src.m); //directly use much faster single-k version (test of single-k skipped above)
		//fw.ePhLoop(k1, k2, DebugEph::ePhProcess, &src); //Call ePh loop to test the single-k stuff above
	}
	
	fw.free();
	FeynWann::finalize();
	return 0;
}
