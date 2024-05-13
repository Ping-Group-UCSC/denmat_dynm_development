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
std::vector<vector3<>> readKpointsFile(string fname){
	std::vector<vector3<>> kArr;
	logPrintf("Reading '%s' ... ", fname.c_str()); logFlush();
	ifstream ifs(fname); if (!ifs.is_open()) die("could not open file.\n");
	while (!ifs.eof()){
		string line; getline(ifs, line);
		trim(line);
		if (!line.length()) continue;
		//Parse line
		istringstream iss(line);
		string key; iss >> key;
		if (key == "kpoint"){
			vector3<> k;
			iss >> k[0] >> k[1] >> k[2];
			kArr.push_back(k);
		}
	}
	ifs.close();
	logPrintf("done.\n");
	return kArr;
}

//Write debug code within process() to examine arbitrary e-ph properties along a k- or q-path
struct DebugDefect{
	int bandStart, bandStop; //optional band range read in from input

	//Previously computed quantities using single-k version to test against transformed ones:
	FeynWann::StateE e1, e2;
	FeynWann::MatrixDefect m;
	bool spinAvailable, layerAvailable;

	DebugDefect(int bandStart, int bandStop)
		: bandStart(bandStart), bandStop(bandStop)
	{}

	void process(const FeynWann::MatrixDefect& mD){
		const diagMatrix& E1 = mD.e1->E;
		const diagMatrix& E2 = mD.e2->E;

		//---- Single k compute debug ----

		//---- e-i matrix element debug ----
		logPrintf("|g_{}(k1,k2)|: ");
		for (int b1 = bandStart; b1 < bandStop; b1++){
			for (int b2 = bandStart; b2 < bandStop; b2++){
				double m2 = 0, ndeg = 0;
				for (int d1 = bandStart; d1 < bandStop; d1++)
				for (int d2 = bandStart; d2 < bandStop; d2++)
				if ((fabs(E1[d1] - E1[b1]) < 1e-6) and(fabs(E2[d2] - E2[b2]) < 1e-6)){
					m2 += mD.M(d1, d2).norm(); ndeg += 1;
				}
				logPrintf("%lg ", sqrt(m2 / ndeg) / eV);
			}
		}
		logPrintf("\n"); logFlush();

		//---- Spin commutator debug ---
		if (spinAvailable){
			matrix S1z = degenerateProject(mD.e1->S[2], E1);
			matrix S2z = degenerateProject(mD.e2->S[2], E2);
			const matrix& G = mD.M;
			matrix SGcomm = S1z * G - G * S2z;
			logPrintf("SGcommDEBUG: ");
			for (int b1 = bandStart; b1 < bandStop; b1++){
				for (int b2 = bandStart; b2 < bandStop; b2++){
					double m2 = 0, ndeg = 0;
					for (int d1 = bandStart; d1 < bandStop; d1++)
					for (int d2 = bandStart; d2 < bandStop; d2++)
					if ((fabs(E1[d1] - E1[b1]) < 1e-6) and(fabs(E2[d2] - E2[b2]) < 1e-6)){
						m2 += SGcomm(d1, d2).norm(); ndeg += 1;
					}
					logPrintf("%lg ", sqrt(m2 / ndeg) / eV);
				}
			}
			logPrintf("\n"); logFlush();
		}
	}
	static void defectProcess(const FeynWann::MatrixDefect& mD, void* params){
		((DebugDefect*)params)->process(mD);
	}

	inline matrix degenerateProject(const matrix& M, const diagMatrix& E){
		static const double degeneracyThreshold = 1e-6;
		matrix out = M;
		complex* outData = out.data();
		for (int b2 = 0; b2<out.nCols(); b2++)
		for (int b1 = 0; b1<out.nRows(); b1++){
			if (fabs(E[b1] - E[b2]) > degeneracyThreshold) (*outData) = 0;
			outData++;
		}
		return out;
	}
};

int main(int argc, char** argv)
{
	InitParams ip = FeynWann::initialize(argc, argv, "Print electron impurity matrix element, |g_q|.");

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(ip.inputFilename);
	const vector3<> k1 = inputMap.getVector("k1");
	string k2file = inputMap.getString("k2file"); //file containing list of k2 points (like a bandstruct.kpoints file)
	int bandStart = inputMap.get("bandStart", 0);
	int bandStop = inputMap.get("bandStop", 0); //replaced with nBands below if 0
	const string defectName = inputMap.getString("defectName");
	FeynWannParams fwp(&inputMap);

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("k1 = "); k1.print(globalLog, " %lg ");
	logPrintf("k2file = %s\n", k2file.c_str());
	logPrintf("bandStart = %d\n", bandStart);
	logPrintf("bandStop = %d\n", bandStop);
	logPrintf("defectName = %s\n", defectName.c_str());
	fwp.printParams();

	//Read k-points:
	std::vector<vector3<>> k2arr = readKpointsFile(k2file);
	logPrintf("Read %lu k-points from '%s'\n", k2arr.size(), k2file.c_str());

	//Initialize FeynWann:
	fwp.needDefect = defectName;
	fwp.needSpin = true;
	FeynWann fw(fwp);
	if (!bandStop) bandStop = fw.nBands;

	if (ip.dryRun){
		logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		fw.free();
		FeynWann::finalize();
		return 0;
	}
	logPrintf("\n");
	DebugDefect src(bandStart, bandStop);
	src.spinAvailable = fwp.needSpin;
	fw.eCalc(k1, src.e1);
	if (mpiGroup->isHead()){
		logPrintf("E1[eV]: ");
		(src.e1.E*(1. / eV)).print(globalLog);
	}
	for (vector3<> k2 : k2arr)
	{	//Compute single k quantities:
		fw.eCalc(k2, src.e2);
		fw.defectCalc(src.e1, src.e2, src.m);

		if (mpiGroup->isHead()) src.process(src.m); //directly use much faster single-k version (test of single-k skipped above)
	}

	fw.free();
	FeynWann::finalize();
	return 0;
}
