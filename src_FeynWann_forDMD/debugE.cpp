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

//Write debug code within process() to examine arbitrary electronic properties along a k-path
struct DebugE
{
	FeynWann::StateE e; //updated for each k before calling process()
	std::vector<diagMatrix> Earr;
	std::vector<matrix> Larr;
	std::vector<matrix> Qarr;
	
	void process()
	{	logPrintf("k: "); e.k.print(globalLog, " %9.6lf ");
		Earr.push_back(e.E);
		for(int iDir=0; iDir<3; iDir++) Larr.push_back(e.L[iDir]);
		for(int iComp=0; iComp<5; iComp++) Qarr.push_back(e.Q[iComp]);
	}
	
	void saveOutputs()
	{	write(Earr, "debug.E");
		write(Larr, "debug.L");
		write(Qarr, "debug.Q");
	}
	
	static void write(const std::vector<diagMatrix>& Marr, string fname)
	{	logPrintf("Dumping '%s' ... ", fname.c_str()); logFlush();
		FILE* fp = fopen(fname.c_str(), "w");
		for(const diagMatrix& M: Marr)
			fwrite(M.data(), sizeof(double), M.nRows(), fp);
		fclose(fp);
		logPrintf("done.\n"); logFlush();
	}

	static void write(const std::vector<matrix>& Marr, string fname)
	{	logPrintf("Dumping '%s' ... ", fname.c_str()); logFlush();
		FILE* fp = fopen(fname.c_str(), "w");
		for(const matrix& M: Marr)
			M.write(fp);
		fclose(fp);
		logPrintf("done.\n"); logFlush();
	}
};

int main(int argc, char** argv)
{	
	InitParams ip = FeynWann::initialize(argc, argv, "Debug electronic matrix elements.");

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(ip.inputFilename);
	string kFile = inputMap.getString("kFile"); //file containing list of k-points (like a bandstruct.kpoints file)
	FeynWannParams fwp(&inputMap);

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("kFile = %s\n", kFile.c_str());
	fwp.printParams();

	//Read k-points:
	std::vector<vector3<>> kArr = readKpointsFile(kFile);
	logPrintf("Read %lu k-points from '%s'\n", kArr.size(), kFile.c_str());
	
	//Initialize FeynWann:
	fwp.needL = true;
	fwp.needQ = true;
	fwp.ePhHeadOnly = true; //so as to debug k-path alone
	FeynWann fw(fwp);
	
	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		fw.free();
		FeynWann::finalize();
		return 0;
	}
	logPrintf("\n");
	
	DebugE de;
	for(vector3<> k: kArr)
	{	fw.eCalc(k, de.e);
		if(mpiGroup->isHead())
			de.process();
	}
	de.saveOutputs();
	fw.free();
	FeynWann::finalize();
	return 0;
}
