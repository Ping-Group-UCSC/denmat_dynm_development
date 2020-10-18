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

#include "FeynWann.h"
#include "InputMap.h"
#include <core/BlasExtra.h>
#include <core/Random.h>
#include <wannier/WannierMinimizer.h>
#include <fftw3-mpi.h>
#include "config.h"

FeynWannParams::FeynWannParams(InputMap* inputMap)
: iSpin(0), totalEprefix("Wannier/totalE"), phononPrefix("Wannier/phonon"), wannierPrefix("Wannier/wannier"),
needSymmetries(false), needPhonons(false), needVelocity(false), needSpin(false),
needLinewidth_ee(false), needLinewidth_ePh(false), needLinewidthP_ePh(false),
ePhHeadOnly(false), maskOptimize(false), EzExt(0.), scissor(0.)
{
	if (inputMap)
	{
		const double nm = 10 * Angstrom;
		const double Tesla = eV*sec / (meter*meter);
		Bext = inputMap->getVector("Bext", vector3<>(0., 0., 0.)) * Tesla;
		EzExt = inputMap->get("EzExt", 0.) * eV / nm;
		scissor = inputMap->get("scissor", 0.) * eV;
	}
}

void FeynWannParams::printParams() const
{
	logPrintf("Bext = "); Bext.print(globalLog, " %lg ");
	logPrintf("EzExt = %lg\n", EzExt);
	logPrintf("scissor = %lg\n", scissor);
}

//Fillings grid on [0,1] for which to calculate e-ph linewidths
inline std::vector<double> getFgrid(int nInterp)
{	std::vector<double> fGrid(nInterp+1);
	double df = 1./nInterp;
	for(int i=0; i<=nInterp; i++)
		fGrid[i] = i*df;
	return fGrid;
}
const std::vector<double> FeynWannParams::fGrid_ePh = getFgrid(4);

InitParams FeynWann::initialize(int argc, char** argv, const char* description)
{	InitParams ip;
	ip.packageName = PACKAGE_NAME;
	ip.versionString = VERSION_STRING;
	ip.versionHash = GIT_HASH;
	ip.description = description;
	initSystemCmdline(argc, argv, ip);
	fftw_mpi_init();
	return ip;
}

void FeynWann::finalize()
{	fftw_mpi_cleanup();
	finalizeSystem();
}

vector3<> FeynWann::randomVector(MPIUtil* mpiUtil)
{	vector3<> v;
	for(int iDir=0; iDir<3; iDir++)
		v[iDir] = Random::uniform();
	if(mpiUtil) mpiUtil->bcast(&v[0], 3);
	return v;
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

std::vector<vector3<>> readArrayVec3(string fname); //Read an array of vector3<> from a plain text file (implemented in wannier/WannierMinimizer_phonon.cpp)


FeynWann::FeynWann(FeynWannParams& fwp)
: fwp(fwp), nAtoms(0), nSpins(0), nSpinor(0), spinWeight(0), isMetal(false), mu(NAN), nElectrons(0), polar(false), inEphLoop(false), 
eEneOnly(false), ePhEstart(-1), ePhEstop(0)
{	
	//Create inter-group communicator if requested:
	std::shared_ptr<MPIUtil> mpiInterGroup;
	const char* envFwSharedRead = getenv("FW_SHARED_READ");
	if(envFwSharedRead and string(envFwSharedRead)=="yes")
	{	int groupSizeMin = mpiGroup->nProcesses();
		int groupSizeMax = mpiGroup->nProcesses();
		mpiWorld->allReduce(groupSizeMin, MPIUtil::ReduceMin);
		mpiWorld->allReduce(groupSizeMax, MPIUtil::ReduceMax);
		if(groupSizeMin != groupSizeMax)
			logPrintf("\nIgnoring FW_SHARED_READ=yes due to non-uniform process grid.\n");
		else
		{	std::vector<int> ranks;
			for(int i=mpiGroup->iProcess(); i<mpiWorld->nProcesses(); i+=mpiGroup->nProcesses())
				ranks.push_back(i);
			mpiInterGroup = std::make_shared<MPIUtil>(mpiWorld, ranks);
			logPrintf("\nFound FW_SHARED_READ=yes and initialized inter-group communicators for shared read.\n");
		}
	}

	//Read relevant parameters from totalE.out:
	string fname = fwp.totalEprefix + ".out";
	logPrintf("\nReading '%s' ... ", fname.c_str()); logFlush();
	ifstream ifs(fname); if(!ifs.is_open()) die("could not open file.\n");
	bool initDone = false; //whether finished reading the initialization part of totalE.out
	int nBandsDFT = 0; //number of DFT bands (>= this->nBands = # Wannier bands)
	int nStatesDFT = 0; //number of reduced k-pts * spins in DFT
	while(!ifs.eof())
	{	string line; getline(ifs, line);
		if(line.find("Initializing the grid") != string::npos)
		{	getline(ifs, line); //skip the line containing "R = "
			for(int j=0; j<3; j++)
			{	getline(ifs, line);
				sscanf(line.c_str(), "[ %lf %lf %lf ]", &R(j,0), &R(j,1), &R(j,2));
			}
			Omega = fabs(det(R));
		}
		else if(line.find("kpoint-folding") != string::npos)
		{	istringstream iss(line); string buf;
			iss >> buf >> kfold[0] >> kfold[1] >> kfold[2];
		}
		else if(line.find("spintype") != string::npos)
		{	istringstream iss(line); string buf, spinString;
			iss >> buf >> spinString;
			if(spinString == "no-spin")
			{	nSpins = 1;
				nSpinor = 1;
			}
			else if(spinString == "z-spin")
			{	nSpins = 2;
				nSpinor = 1;
			}
			else //non-collinear modes
			{	nSpins = 1;
				nSpinor = 2;
			}
			spinWeight = 2/(nSpins*nSpinor);
			if(fwp.iSpin<0 || fwp.iSpin>=nSpins)
				die("iSpin = %d not in interval [0,nSpins), where nSpins = %d for this system.\n\n", fwp.iSpin, nSpins);
			spinSuffix = (nSpins==1 ? "" : (fwp.iSpin==0 ? "Up" : "Dn"));
		}
		else if(line.find("coulomb-interaction") != string::npos)
		{	istringstream iss(line); string cmdName, typeString, dirString;
			iss >> cmdName >> typeString >> dirString;
			if(typeString == "Periodic")
			{	isTruncated = vector3<bool>(false, false, false);
			}
			else if(typeString == "Slab")
			{	isTruncated = vector3<bool>(false, false, false);
				if(dirString == "100") isTruncated[0] = true;
				else if(dirString == "010") isTruncated[1] = true;
				else if(dirString == "001") isTruncated[2] = true;
				else die("Unrecognized truncation direction '%s'\n", dirString.c_str());
			}
			else if(typeString == "Wire" || typeString == "Cylindrical")
			{	isTruncated = vector3<bool>(true, true, true);
				if(dirString == "100") isTruncated[0] = false;
				else if(dirString == "010") isTruncated[1] = false;
				else if(dirString == "001") isTruncated[2] = false;
				else die("Unrecognized truncation direction '%s'\n", dirString.c_str());
			}
			else if(typeString == "Isolated" || typeString == "Spherical")
			{	isTruncated = vector3<bool>(true, true, true);
			}
			else die("Unrecognized truncation type '%s'\n", typeString.c_str());
		}
		else if(line.find("Initialization completed") == 0)
		{	initDone = true;
		}
		else if(initDone && (line.find("FillingsUpdate:") != string::npos))
		{	istringstream iss(line); string buf;
			iss >> buf >> buf >> mu >> buf >> nElectrons;
		}
		else if(initDone && (line.find("# Ionic positions in") != string::npos))
		{	atpos.clear(); //read last version (if many ionic steps in totalE.out)
			atNames.clear();
			bool cartesian = (line.find("cartesian") != string::npos);
			while(true)
			{	getline(ifs, line);
				istringstream iss(line);
				string cmd, atName; vector3<> x;
				iss >> cmd >> atName >> x[0] >> x[1] >> x[2]; //rest (move flag etc. not needed)
				if(cmd != "ion") break;
				if(cartesian) x = inv(R) * x; //convert to lattice
				atpos.push_back(x);
				atNames.push_back(atName);
			}
		}
		else if(line.find("nElectrons:") == 0) //nElectrons, nBands, nStates line
		{	istringstream iss(line); string buf;
			iss >> buf >> nElectrons >> buf >> nBandsDFT >> buf >> nStatesDFT;
		}
	}
	ifs.close();
	logPrintf("done.\n"); logFlush();
	if(!nSpins)
		die("Could not determine spin configuration from DFT output file.");
	if(std::isnan(mu))
	{	logPrintf("NOTE: mu unavailable; assuming semiconductor/insulator and setting to VBM.\n");
		isMetal = false;
		int nValence = int(round(nElectrons/(nSpins*spinWeight))); //number of valence bands
		if(fabs(nValence*nSpins*spinWeight-nElectrons > 1e-6))
			die("Number of electrons incompatible with semiconductor / insulator.\n");
		//Read DFT eigenvalues file:
		ManagedArray<double> Edft; Edft.init(nBandsDFT*nStatesDFT);
		fname = fwp.totalEprefix + ".eigenvals";
		logPrintf("Reading '%s' ... ", fname.c_str()); logFlush();
		Edft.read(fname.c_str());
		logPrintf("done.\n");
		//Find VBM:
		mu = -DBL_MAX;
		for(int q=0; q<nStatesDFT; q++)
			mu = std::max(mu, Edft.data()[q*nBandsDFT+nValence-1]); //highest valence eigenvalue at each q
	}
	else
		isMetal = true;
	logPrintf("mu = %lg\n", mu);
	logPrintf("nElectrons = %lg\n", nElectrons);
	logPrintf("nBandsDFT = %d\n", nBandsDFT);
	logPrintf("nSpins = %d\n", nSpins);
	logPrintf("nSpinor = %d\n", nSpinor);
	logPrintf("spinSuffix = '%s'\n", spinSuffix.c_str());
	logPrintf("kfold = "); kfold.print(globalLog, " %d ");
	logPrintf("isTruncated = "); isTruncated.print(globalLog, " %d ");
	logPrintf("R:\n");
	R.print(globalLog, " %lg ");
	logPrintf("Atoms with fractional coordinates:\n");
	for(unsigned i=0; i<atpos.size(); i++)
		logPrintf("\t%2s %19.15lf %19.15lf %19.15lf\n",
			atNames[i].c_str(), atpos[i][0], atpos[i][1], atpos[i][2]);
	nAtoms = int(atpos.size());
	logPrintf("\n");
	
	//Read symmetries if required
	if(fwp.needSymmetries)
	{	fname = fwp.totalEprefix + ".sym";
		logPrintf("Reading '%s' ... ", fname.c_str()); logFlush();
		ifs.open(fname); if(!ifs.is_open()) die("could not open file.\n");
		sym.clear();
		while(!ifs.eof())
		{	SpaceGroupOp op;
			for(int i=0; i<3; i++) for(int j=0; j<3; j++) ifs >> op.rot(i,j); //rotation
			for(int i=0; i<3; i++) ifs >> op.a[i]; //translation
			if(ifs.good()) sym.push_back(op);
		}
		ifs.close();
		logPrintf("done. Read %lu symmetries.\n", sym.size());
	}
	
	//Read cell map
	cellMap = readCellMap(fwp.wannierPrefix + ".mlwfCellMap" + spinSuffix);
	
	//Find number of wannier centers from Wannier band contrib file:
	{	fname = fwp.wannierPrefix + ".mlwfBandContrib" + spinSuffix;
		logPrintf("Reading '%s' ... ", fname.c_str()); logFlush();
		FILE* fp = fopen(fname.c_str(), "r");
		if(!fp) die("could not open file.\n");
		nBands = 0; //number of Wannier centers
		while(!feof(fp))
		{	int nMin_b, nMax_b; double eMin_b, eMax_b;
			if(fscanf(fp, "%d %d %lf %lf", &nMin_b, &nMax_b, &eMin_b, &eMax_b) != 4) break;
			nBands = std::max(nBands, nMax_b+1); //number of Wannier centers
		}
		fclose(fp);
		logPrintf("done.\n");
		assert(nBands <= nBandsDFT);
	}
	logPrintf("nBands = %d\n", nBands);
	logPrintf("\n");
	
	//Read cell weights:
	fname = fwp.wannierPrefix + ".mlwfCellWeights" + spinSuffix;
	logPrintf("Reading '%s' ... ", fname.c_str()); logFlush();
	cellWeights.init(nBands*nBands, cellMap.size());
	cellWeights.read_real(fname.c_str());
	//--- split to matrix per cell:
	std::vector<matrix> cellWeightsVec(cellMap.size());
	for(size_t iCell=0; iCell<cellWeightsVec.size(); iCell++)
	{	cellWeightsVec[iCell] = cellWeights(0,nBands*nBands, iCell,iCell+1);
		cellWeightsVec[iCell].reshape(nBands,nBands);
	}
	logPrintf("done.\n");
	
	//Read wannier hamiltonian
	realPartOnly = (nSpinor==1);
	fname = fwp.wannierPrefix + ".mlwfH" + spinSuffix;
	Hw = std::make_shared<DistributedMatrix>(fname, realPartOnly,
		mpiGroup, nBands*nBands, cellMap, kfold, false, mpiInterGroup, &cellWeightsVec);
	
	if(fwp.needPhonons)
	{	//Read relevant parameters from phonon.out:
		fname = fwp.phononPrefix + ".out";
		logPrintf("\nReading '%s' ... ", fname.c_str()); logFlush();
		ifs.open(fname); if(!ifs.is_open()) die("could not open file.\n");
		nModes = 0;
		while(!ifs.eof())
		{	string line; getline(ifs, line);
			if(line.find("phonon  \\") != string::npos)
			{	//at start of phonon command print
				string key;
				while(key!="supercell" && (!ifs.eof()))
					ifs >> key; //search for supercell keyword
				ifs >> phononSup[0] >> phononSup[1] >> phononSup[2];
				if(!ifs.good()) die("Failed to read phonon supercell dimensions.\n");
			}
			string cmdName; istringstream(line) >> cmdName;
			if(cmdName == "ion")
				nModes += 3; //3 modes per atom in unit cell
			if(line.find("Unit cell calculation") != string::npos)
				break; //don't need anything else after this from phonon.out
		}
		ifs.close();
		if(!phononSup.length_squared()) die("Failed to read phonon supercell dimensions.\n");
		if(nModes != 3*nAtoms) die("Number of modes = %d in phonon.out is inconsistent with nAtoms = %d in totalE.out\n", nModes, nAtoms);
		logPrintf("done.\n"); logFlush();
		logPrintf("nModes = %d\n", nModes);
		logPrintf("phononSup = "); phononSup.print(globalLog, " %d ");
		for(int iDir=0; iDir<3; iDir++)
		{	kfoldSup[iDir] = kfold[iDir] / phononSup[iDir];
			if(kfoldSup[iDir] * phononSup[iDir] != kfold[iDir])
				die("kfold is not a multiple of phononSup.\n");
		}
		logPrintf("\n");
		
		//Read phonon basis:
		invsqrtM = readPhononBasis(fwp.totalEprefix + ".phononBasis");
		
		//Read phonon cell map:
		fname = fwp.totalEprefix + ".phononCellMap";
		if(fileSize((fname + "Corr").c_str()) > 0) //corrected force matrix cell map exists
			fname += "Corr";
		phononCellMap = readCellMap(fname);
		
		//Read phonon force matrix
		fname = fwp.totalEprefix + ".phononOmegaSq";
		if(fileSize((fname + "Corr").c_str()) > 0) //corrected force matrix exists
			fname += "Corr";
		OsqW = std::make_shared<DistributedMatrix>(fname, true, //phonon omegaSq is always real
			mpiGroup, nModes*nModes, phononCellMap, phononSup, false, mpiInterGroup);
		
		//Read cell maps for electron-phonon matrix elements and sum rule:
		ePhCellMap = readCellMap(fwp.wannierPrefix + ".mlwfCellMapPh" + spinSuffix);
		ePhCellMapSum = readCellMap(fwp.wannierPrefix + ".mlwfCellMapPhSum" + spinSuffix);

		//Read e-ph cell weights:
		std::vector<matrix> ePhCellWeights; //atom x band weights for each cell in ePhCellMap
		{	fname = fwp.wannierPrefix + ".mlwfCellWeightsPh" + spinSuffix;
			logPrintf("Reading '%s' ... ", fname.c_str()); logFlush();
			matrix ePhCellWeightsAll(nAtoms*nBands, ePhCellMap.size());
			ePhCellWeightsAll.read_real(fname.c_str());
			//--- split to matrix per cell:
			ePhCellWeights.resize(ePhCellMap.size());
			for(size_t iCell=0; iCell<ePhCellWeights.size(); iCell++)
			{	ePhCellWeights[iCell] = ePhCellWeightsAll(0,nAtoms*nBands, iCell,iCell+1);
				ePhCellWeights[iCell].reshape(nAtoms,nBands);
			}
			logPrintf("done.\n");
		}
		
		//Read electron-phonon matrix elements
		fname = fwp.wannierPrefix + ".mlwfHePh" + spinSuffix;
		HePhW = std::make_shared<DistributedMatrix>(fname, realPartOnly,
			mpiGroup, nModes*nBands*nBands, ePhCellMap, phononSup, true, mpiInterGroup, &ePhCellWeights);
		
		//Read electron-phonon matrix element sum rule
		fname = fwp.wannierPrefix + ".mlwfHePhSum" + spinSuffix;
		HePhSumW = std::make_shared<DistributedMatrix>(fname, realPartOnly,
			mpiGroup, 3*nBands*nBands, ePhCellMapSum, kfold, false, mpiInterGroup);
		
		//Read gradient matrix element for e-ph sum rule
		fname = fwp.wannierPrefix + ".mlwfD" + spinSuffix;
		Dw = std::make_shared<DistributedMatrix>(fname, realPartOnly,
			mpiGroup, 3*nBands*nBands, cellMap, kfold, false, mpiInterGroup, &cellWeightsVec);
		
		//Check for polarity:
		fname = fwp.wannierPrefix + ".out";
		logPrintf("\nReading '%s' ... ", fname.c_str()); logFlush();
		ifs.open(fname); if(!ifs.is_open()) die("could not open file.\n");
		while(!ifs.eof())
		{	string line; getline(ifs, line);
			if(line.find("wannier  \\") != string::npos)
			{	//at start of wannier command print
				string key, val;
				while(key!="polar" and (!ifs.eof()))
					ifs >> key; //search for polar keyword
				ifs >> val;
				if(ifs.good() and val=="yes")
				{	polar = true;
					break;
				}
			}
		}
		ifs.close();
		logPrintf("done.\n");
		logPrintf("polar = %s\n\n", polar ? "yes" : "no");
		
		if(polar)
		{	//Read Born effective charges:
			Zeff = readArrayVec3(fwp.totalEprefix + ".Zeff");
			//Read optical dielectric tensor:
			std::vector<vector3<>> eps = readArrayVec3(fwp.totalEprefix + ".epsInf");
			omegaEff = Omega;
			truncDir = 3;
			for (int iDir=0; iDir<3; iDir++) 
			{ 	if(isTruncated[iDir]) 
				{	truncDir = iDir;
					omegaEff /= fabs(R(iDir, iDir));
				}
			}
			if (truncDir < 3) 
			{	epsInf2D = eps;
				lrs2D = std::make_shared<LongRangeSum2D>(R, epsInf2D, truncDir);
			}else
			{	epsInf.set_rows(eps[0], eps[1], eps[2]);
				lrs = std::make_shared<LongRangeSum>(R, epsInf);
			}
			//Read cell weights:
			fname = fwp.totalEprefix + ".phononCellWeights";
			logPrintf("Reading '%s' ... ", fname.c_str()); logFlush();
			phononCellWeights.init(nAtoms*nAtoms, phononCellMap.size());
			phononCellWeights.read_real(fname.c_str());
			logPrintf("done.\n");
		}
	}
	
	//Velocity matrix elements
	if(fwp.needVelocity)
	{	fname = fwp.wannierPrefix + ".mlwfP" + spinSuffix;
		Pw = std::make_shared<DistributedMatrix>(fname, realPartOnly,
			mpiGroup, 3*nBands*nBands, cellMap, kfold, false, mpiInterGroup, &cellWeightsVec);
	}
	//Spin matrix elements
	if(not isRelativistic()) fwp.needSpin = false; //spin only available in relatvistic mode
	if(fwp.needSpin)
	{	fname = fwp.wannierPrefix + ".mlwfS" + spinSuffix;
		Sw = std::make_shared<DistributedMatrix>(fname, realPartOnly,
			mpiGroup, 3*nBands*nBands, cellMap, kfold, false, mpiInterGroup, &cellWeightsVec);
	}
	//z position matrix elements
	if (fwp.EzExt)
	{
		fname = fwp.wannierPrefix + ".mlwfZ" + spinSuffix;
		Zw = std::make_shared<DistributedMatrix>(fname, realPartOnly,
			mpiGroup, nBands*nBands, cellMap, kfold, false, mpiInterGroup, &cellWeightsVec);
	}
	//Linewidths:
	if(fwp.needLinewidth_ee)
	{	//e-e:
		fname = fwp.wannierPrefix + ".mlwfImSigma_ee" + spinSuffix;
		ImSigma_eeW = std::make_shared<DistributedMatrix>(fname, realPartOnly,
			mpiGroup, nBands*nBands, cellMap, kfold, false, mpiInterGroup, &cellWeightsVec);
	}
	if(fwp.needLinewidth_ePh)
	{	//e-ph:
		fname = fwp.wannierPrefix + ".mlwfImSigma_ePh" + spinSuffix;
		ImSigma_ePhW = std::make_shared<DistributedMatrix>(fname, realPartOnly,
			mpiGroup, nBands*nBands*FeynWannParams::fGrid_ePh.size(), cellMap, kfold, false, mpiInterGroup, &cellWeightsVec);
	}
	if(fwp.needLinewidthP_ePh)
	{	//e-ph:
		fname = fwp.wannierPrefix + ".mlwfImSigmaP_ePh" + spinSuffix;
		ImSigmaP_ePhW = std::make_shared<DistributedMatrix>(fname, realPartOnly,
			mpiGroup, nBands*nBands*FeynWannParams::fGrid_ePh.size(), cellMap, kfold, false, mpiInterGroup, &cellWeightsVec);
	}
	
	logPrintf("\n");
}

void FeynWann::free()
{	Hw = 0;
	Pw = 0;
	Sw = 0;
	ImSigma_eeW = 0;
	ImSigma_ePhW = 0;
	ImSigmaP_ePhW = 0;
	OsqW = 0;
	HePhW = 0;
	HePhSumW = 0;
	Dw = 0;
}

//Get iMatrix'th matrix of specified dimensions from pointer src, assuming they are stored contiguously there in column-major order)
inline matrix getMatrix(const complex* src, int nRows, int nCols, int iMatrix=0)
{	matrix result(nRows, nCols);
	eblas_copy(result.data(), src + iMatrix*result.nData(), result.nData());
	return result;
}

//Prepare and broadcast matrices on custom communicator:
inline void bcast(diagMatrix& m, int nRows, MPIUtil* mpiUtil, int root)
{	m.resize(nRows);
	mpiUtil->bcast(m.data(), nRows, root);
}
inline void bcast(matrix& m, int nRows, int nCols, MPIUtil* mpiUtil, int root)
{	if(!m) m.init(nRows, nCols);
	mpiUtil->bcast(m.data(), m.nData(), root);
}

template<typename T> vector3<T> elemwiseProd(vector3<int> a, vector3<T> b)
{	return vector3<T>(a[0]*b[0], a[1]*b[1], a[2]*b[2]);
}

//Loop i till iStop, sampling a 3D mesh of dimensions S
//At each point, set fractional coordinates x offset by x0 and run code
#define PartialLoop3D(S, i, iStop, x, x0, code) \
	vector3<int> i##v( \
		i / (S[2]*S[1]), \
		(i/S[2]) % S[1], \
		i % S[2] ); \
	vector3<> i##Frac(1./S[0], 1./S[1], 1./S[2]); \
	while(i<iStop) \
	{	\
		x = x0 + vector3<>(i##v[0]*i##Frac[0], i##v[1]*i##Frac[1], i##v[2]*i##Frac[2]); \
		code \
		\
		i++; if(i==iStop) break; \
		i##v[2]++; \
		if(i##v[2]==S[2]) \
		{	i##v[2]=0; \
			i##v[1]++; \
			if(i##v[1]==S[1]) \
			{	i##v[1] = 0; \
				i##v[0]++; \
			} \
		} \
	}

void FeynWann::eLoop(const vector3<>& k0, FeynWann::eProcessFunc eProcess, void* params)
{	static StopWatch watchCallback("FeynWann::eLoop:callback");
	//Run Fourier transforms with this offset:
	Hw->transform(k0);
	if(fwp.needVelocity) Pw->transform(k0);
	if(fwp.needSpin) Sw->transform(k0);
	if (fwp.EzExt) Zw->transform(k0);
	if (fwp.needLinewidth_ee) ImSigma_eeW->transform(k0);
	if(fwp.needLinewidth_ePh) ImSigma_ePhW->transform(k0);
	if(fwp.needLinewidthP_ePh) ImSigmaP_ePhW->transform(k0);
	//Call eProcess for k-points on present process:
	int ik = Hw->ikStart;
	int ikStop = ik + Hw->nk;
	StateE state;
	PartialLoop3D(kfold, ik, ikStop, state.k, k0,
		state.ik = ik;
		setState(state);
		watchCallback.start();
		eProcess(state, params);
		watchCallback.stop();
	)
}
void FeynWann::eCalc(const vector3<>& k, FeynWann::StateE& e)
{	//Compute Fourier versions for this k:
	Hw->compute(k);
	if (fwp.needVelocity) Pw->compute(k);
	if (fwp.needSpin) Sw->compute(k);
	if (fwp.EzExt) Zw->compute(k);
	if(fwp.needLinewidth_ee) ImSigma_eeW->compute(k);
	if(fwp.needLinewidth_ePh) ImSigma_ePhW->compute(k);
	if(fwp.needLinewidthP_ePh) ImSigmaP_ePhW->compute(k);
	if(fwp.needPhonons) //prepare sum rule quantities
	{	inEphLoop = true;
		Dw->compute(k);
		HePhSumW->compute(k);
	}
	//Prepare state on group head:
	e.ik = 0;
	e.k = k;
	if(mpiGroup->isHead()) setState(e);
	inEphLoop = false;
}


void FeynWann::phLoop(const vector3<>& q0, FeynWann::phProcessFunc phProcess, void* params)
{	static StopWatch watchCallback("FeynWann::phLoop:callback");
	assert(fwp.needPhonons);
	//Run Fourier transforms with this offset:
	OsqW->transform(q0);
	//Call phProcess for q-points on present process:
	int iq = OsqW->ikStart;
	int iqStop = iq + OsqW->nk;
	StatePh state;
	PartialLoop3D(phononSup, iq, iqStop, state.q, q0,
		state.iq = iq;
		state.iqFine = 0; //Only meaningful in ePhLoop
		setState(state);
		watchCallback.start();
		phProcess(state, params);
		watchCallback.stop();
	)
}
void FeynWann::phCalc(const vector3<>& q, FeynWann::StatePh& ph)
{	assert(fwp.needPhonons);
	//Compute Fourier versions for this q:
	OsqW->compute(q);
	//Prepare state on group head:
	ph.iqFine = 0;
	ph.iq = 0;
	ph.q = q;
	if(mpiGroup->isHead()) setState(ph);
}


void FeynWann::ePhLoop(const vector3<>& k01, const vector3<>& k02, FeynWann::ePhProcessFunc ePhProcess, void* params)
{	static StopWatch watchBcast("FeynWann::ePhLoop:bcast"); 
	static StopWatch watchCallback("FeynWann::ePhLoop:callback");
	assert(fwp.needPhonons);
	int prodKfold = Hw->nkTot;
	int prodSup = OsqW->nkTot;
	int prodSupSq = HePhW->nkTot;
	assert(prodSupSq == prodSup*prodSup);
	//Initialize electronic states for 1 and 2:
	#define PrepareElecStates(i) \
		bool withinRange##i = false; \
		std::vector<StateE> e##i(prodKfold); /* States */ \
		{	Hw->transform(k0##i); \
			if (fwp.needVelocity) Pw->transform(k0##i); \
			if (fwp.needSpin) Sw->transform(k0##i); \
			if (fwp.EzExt) Zw->transform(k0##i); \
			if(fwp.needLinewidth_ee) ImSigma_eeW->transform(k0##i); \
			if(fwp.needLinewidth_ePh) ImSigma_ePhW->transform(k0##i); \
			if(fwp.needLinewidthP_ePh) ImSigmaP_ePhW->transform(k0##i); \
			HePhSumW->transform(k0##i); \
			Dw->transform(k0##i); \
			int ik = Hw->ikStart; \
			int ikStop = ik + Hw->nk; \
			PartialLoop3D(kfold, ik, ikStop, e##i[ik].k, k0##i, \
				e##i[ik].ik = ik; \
				setState(e##i[ik]); \
				if (e##i[ik].withinRange) withinRange##i = true; \
			) \
			/* Make available on all processes of group */ \
			if(mpiGroup->nProcesses() > 1) \
			{	watchBcast.start(); \
				for(int whose=0; whose<mpiGroup->nProcesses(); whose++) \
					for(int ik=Hw->ikStartProc[whose]; ik<Hw->ikStartProc[whose+1]; ik++) \
						bcastState(e##i[ik], mpiGroup, whose); \
				mpiGroup->allReduce(withinRange##i, MPIUtil::ReduceLOr); \
				watchBcast.stop(); \
			} \
		}
	inEphLoop = true; //turns on sum rule handling in setState and bcastState
	PrepareElecStates(1) //prepares e1 and V1
	if (not withinRange1) return; //no states in active window of 1
	PrepareElecStates(2) //prepares e2 and V2
	if (not withinRange2) return; //no states in active window of 2
	inEphLoop = false;
	#undef PrepareElecStates
	//Loop over phonon q offsets:
	vector3<> kfoldInv(1./kfold[0], 1./kfold[1], 1./kfold[2]);
	vector3<int> iqSup;
	for(iqSup[0]=0; iqSup[0]<kfoldSup[0]; iqSup[0]++)
	for(iqSup[1]=0; iqSup[1]<kfoldSup[1]; iqSup[1]++)
	for(iqSup[2]=0; iqSup[2]<kfoldSup[2]; iqSup[2]++)
	{	if(fwp.ePhHeadOnly and iqSup.length_squared()) continue; //k-path debug mode
		//Prepare phonon states:
		vector3<> q0 = k01 - k02 + elemwiseProd(iqSup, kfoldInv);
		OsqW->transform(q0);
		std::vector<StatePh> ph(prodSup);
		{	int iq = OsqW->ikStart;
			int iqStop = iq + OsqW->nk;
			PartialLoop3D(phononSup, iq, iqStop, ph[iq].q, q0,
				ph[iq].iq = iq;
				ph[iq].iqFine = calculateIndex(iqSup + elemwiseProd(kfoldSup, iqv), kfold);
				setState(ph[iq]);
			)
			//Make available on all processes of group:
			if(mpiGroup->nProcesses() > 1)
			{	watchBcast.start();
				for(int whose=0; whose<mpiGroup->nProcesses(); whose++)
					for(int iq=OsqW->ikStartProc[whose]; iq<OsqW->ikStartProc[whose+1]; iq++)
						bcastState(ph[iq], mpiGroup, whose);
				watchBcast.stop();
			}
		}
		//Loop over k2 supercell offsets:
		vector3<int> ik2sup;
		for(ik2sup[0]=0; ik2sup[0]<kfoldSup[0]; ik2sup[0]++)
		for(ik2sup[1]=0; ik2sup[1]<kfoldSup[1]; ik2sup[1]++)
		for(ik2sup[2]=0; ik2sup[2]<kfoldSup[2]; ik2sup[2]++)
		{	vector3<> k02cur = k02 + elemwiseProd(ik2sup, kfoldInv);
			vector3<int> ik1sup = iqSup + ik2sup; //momentum conservation
			vector3<> k01cur = q0 + k02cur; //momentum conservation
			if(fwp.ePhHeadOnly and ik2sup.length_squared()) continue; //k-path debug mode
			//Energy window
			withinRange1 = false; withinRange2 = false;
			if (ePhEstart < ePhEstop){
				int ik1 = 0; vector3<> k1;
				PartialLoop3D(phononSup, ik1, prodSup, k1, k01cur,
					int ik1net = calculateIndex(ik1sup + elemwiseProd(kfoldSup, ik1v), kfold);
					if (e1[ik1net].withinRange) { withinRange1 = true; break; })
				if (withinRange1){
					int ik2 = 0; vector3<> k2;
					PartialLoop3D(phononSup, ik2, prodSup, k2, k02cur,
						int ik2net = calculateIndex(ik2sup + elemwiseProd(kfoldSup, ik2v), kfold);
						if (e2[ik2net].withinRange) { withinRange2 = true; break; })
				}
			}
			if (!withinRange1 || !withinRange2) continue;
			//Calculate electron-phonon matrix elements:
			HePhW->transform(k01cur, k02cur);
			int ikPair = 0;
			int ikPairStart = HePhW->ikStart;
			int ikPairStop = ikPairStart + HePhW->nk;
			int ik1 = 0; vector3<> k1;
			PartialLoop3D(phononSup, ik1, prodSup, k1, k01cur,
				//Identify associated electronic state
				int ik1net = calculateIndex(ik1sup + elemwiseProd(kfoldSup, ik1v), kfold);
				int ik2 = 0; vector3<> k2;
				PartialLoop3D(phononSup, ik2, prodSup, k2, k02cur,
					if(ikPair>=ikPairStart and ikPair<ikPairStop //subset to be evaluated on this process
						and (not (fwp.ePhHeadOnly and ikPair)) ) //overridden in k-path debug mode to be ikPair==0 alone
					{	//Identify associated electronic and phonon states:
						int ik2net = calculateIndex(ik2sup + elemwiseProd(kfoldSup, ik2v), kfold);
						if (e1[ik1net].withinRange && e2[ik2net].withinRange){
							int iqIndex = calculateIndex(ik1v - ik2v, phononSup);
							//Set e-ph matrix elements:
							MatrixEph m;
							setMatrix(e1[ik1net], e2[ik2net], ph[iqIndex], ikPair, m);
							//Invoke call-back function:
							watchCallback.start();
							ePhProcess(m, params);
							watchCallback.stop();
						}
					}
					ikPair++;
				)
			)
		}
	}
}
void FeynWann::ePhCalc(const FeynWann::StateE& e1, const FeynWann::StateE& e2, const FeynWann::StatePh& ph, FeynWann::MatrixEph& m)
{	assert(fwp.needPhonons);
	assert(circDistanceSquared(e1.k-e2.k, ph.q) < 1e-8);
	//Compute Fourier version of HePh for specified k1,k2 pair:
	HePhW->compute(e1.k, e2.k);
	//Prepare state on group head:
	if(mpiGroup->isHead()) setMatrix(e1, e2, ph, 0, m);
}

void FeynWann::symmetrize(matrix3<>& m) const
{	matrix3<> mOut;
	matrix3<> invR = inv(R);
	int nSym = 0;
	for (const SpaceGroupOp& op : sym)
	{	matrix3<> rot = R * op.rot * invR; //convert to Cartesian
		//Exclude rotations that don't leave fields invariant
		if (fwp.Bext.length_squared())
		{	if ((fwp.Bext - rot*fwp.Bext).length() > symmThreshold)
				continue;
		}
		if (fwp.EzExt)
		{	vector3<> Eext(0., 0., fwp.EzExt);
			if ((Eext - rot*Eext).length() > symmThreshold)
				continue;
		}
		mOut += rot * m * (~rot);
		nSym++;
	}
	m = mOut * (1. / nSym);
	//Set near-zero to exact zero:
	double mCut = 1e-14*sqrt(trace((~m)*m));
	for (int i = 0; i<3; i++)
	for (int j = 0; j<3; j++)
	if (fabs(m(i, j)) < mCut)
		m(i, j) = 0.;
}

void FeynWann::setState(FeynWann::StateE& state)
{	static StopWatch watchRotations("FeynWann::setState:rotations");
	//Get and diagonalize Hamiltonian:
	matrix Hk = getMatrix(Hw->getResult(state.ik), nBands, nBands);
	if (fwp.needSpin and fwp.Bext.length_squared())
	{	//Add Zeeman perturbation:
		for (int iDir = 0; iDir<3; iDir++)
		if (fwp.Bext[iDir]) Hk += 0.5 * fwp.Bext[iDir] * getMatrix(Sw->getResult(state.ik), nBands, nBands, iDir);
	}
	if (fwp.EzExt) //Add Stark perturbation:
		Hk += fwp.EzExt * getMatrix(Zw->getResult(state.ik), nBands, nBands);
	Hk.diagonalize(state.U, state.E);
	for(double& E: state.E) E -= mu; //reference to Fermi level
	if (fwp.scissor)
	{	//Apply scissor operator (move up unoccupied states):
		for (double& E : state.E)
		if (E > 1e-3)
			E += fwp.scissor;
	}
	if (eEneOnly) return;
	//Check whether any states in range (only if not already masked out by initial value of withinRange):
	state.withinRange = true;
	if (inEphLoop and (ePhEstart < ePhEstop)){
		state.withinRange = false;
		for (double& E : state.E)
		if (E >= ePhEstart and E <= ePhEstop){
			state.withinRange = true; break;
		}
	}
	if (not state.withinRange) return; //Remaining quantities will never be used
	watchRotations.start();
	//Velcoity matrix, if needed:
	if(fwp.needVelocity)
	{	state.vVec.resize(nBands);
		for(int iDir=0; iDir<3; iDir++)
		{	state.v[iDir] = complex(0,-1) //Since P was stored with -i omitted (to make it real when possible)
				* (dagger(state.U) * getMatrix(Pw->getResult(state.ik), nBands, nBands, iDir) * state.U);
			//Extract diagonal parts for convenience:
			for(int b=0; b<nBands; b++)
				state.vVec[b][iDir] = state.v[iDir](b,b).real();
		}
	}
	if(fwp.needSpin)
	{	state.Svec.resize(nBands);
		for(int iDir=0; iDir<3; iDir++)
		{	state.S[iDir] = dagger(state.U) * getMatrix(Sw->getResult(state.ik), nBands, nBands, iDir) * state.U;
			//Extract diagonal parts for convenience:
			for(int b=0; b<nBands; b++)
				state.Svec[b][iDir] = state.S[iDir](b,b).real();
		}
	}
	//Linewidths, as needed:
	if(fwp.needLinewidth_ee)
		state.ImSigma_ee = diag(dagger(state.U) * getMatrix(ImSigma_eeW->getResult(state.ik), nBands, nBands) * state.U);
	if(fwp.needLinewidth_ePh)
	{	state.logImSigma_ePhArr.resize(FeynWannParams::fGrid_ePh.size());
		for(unsigned iMat=0; iMat<state.logImSigma_ePhArr.size(); iMat++)
			state.logImSigma_ePhArr[iMat] = diag(dagger(state.U) * getMatrix(ImSigma_ePhW->getResult(state.ik), nBands, nBands, iMat) * state.U);
	}
	if(fwp.needLinewidthP_ePh)
	{	state.logImSigmaP_ePhArr.resize(FeynWannParams::fGrid_ePh.size());
		for(unsigned iMat=0; iMat<state.logImSigmaP_ePhArr.size(); iMat++)
			state.logImSigmaP_ePhArr[iMat] = diag(dagger(state.U) * getMatrix(ImSigmaP_ePhW->getResult(state.ik), nBands, nBands, iMat) * state.U);
	}
	//e-ph sum rule if needed:
	if(inEphLoop)
	{	state.dHePhSum.init(nBands*nBands, 3);
		complex* dHsumData = state.dHePhSum.dataPref();
		for(int iDir=0; iDir<3; iDir++)
		{	matrix D = dagger(state.U) * getMatrix(Dw->getResult(state.ik), nBands, nBands, iDir) * state.U;
			matrix H = dagger(state.U) * getMatrix(HePhSumW->getResult(state.ik), nBands, nBands, iDir) * state.U;
			//Compute error in the sum rule:
			const double Emag = 1e-3; //damp correction for energy differences >> Emag (to handle fringes of Wannier window)
			const double expFac = -1./(Emag*Emag);
			complex* Hdata = H.data();
			const complex* Ddata = D.data();
			for(int b2=0; b2<nBands; b2++)
				for(int b1=0; b1<nBands; b1++)
				{	double E12 = state.E[b1] - state.E[b2];
					*Hdata -= (*(Ddata++)) * E12;
					*Hdata *= exp(expFac*E12*E12); //damp correction based on energy difference
					Hdata++;
				}
			//Rotate back to Wannier basis and store to HePhSum
			H = state.U * H * dagger(state.U);
			callPref(eblas_copy)(dHsumData, H.data(), H.nData());
			dHsumData += H.nData();
		}
	}
	watchRotations.stop();
}

void FeynWann::bcastState(FeynWann::StateE& state, MPIUtil* mpiUtil, int root)
{	if(mpiUtil->nProcesses()==1) return; //no communictaion needed
	mpiUtil->bcast(state.ik, root);
	mpiUtil->bcast(&state.k[0], 3, root);
	//Energy and eigenvectors:
	bcast(state.E, nBands, mpiUtil, root);
	mpiUtil->bcast(state.withinRange, root);
	if (not state.withinRange) return; //Remaining quantities will never be used
	bcast(state.U, nBands, nBands, mpiUtil, root);
	//Velocity matrix, if needed:
	if(fwp.needVelocity)
	{	for(int iDir=0; iDir<3; iDir++)
			bcast(state.v[iDir], nBands, nBands, mpiUtil, root);
		state.vVec.resize(nBands);
		mpiUtil->bcastData(state.vVec, root);
	}
	//Spin matrix, if needed:
	if(fwp.needSpin)
	{	for(int iDir=0; iDir<3; iDir++)
			bcast(state.S[iDir], nBands, nBands, mpiUtil, root);
		state.Svec.resize(nBands);
		mpiUtil->bcastData(state.Svec, root);
	}
	//Linewidths, if needed:
	if(fwp.needLinewidth_ee) bcast(state.ImSigma_ee, nBands, mpiUtil, root);
	if(fwp.needLinewidth_ePh)
	{	state.logImSigma_ePhArr.resize(FeynWannParams::fGrid_ePh.size());
		for(diagMatrix& d: state.logImSigma_ePhArr) bcast(d, nBands, mpiUtil, root);
	}
	if(fwp.needLinewidthP_ePh)
	{	state.logImSigmaP_ePhArr.resize(FeynWannParams::fGrid_ePh.size());
		for(diagMatrix& d: state.logImSigmaP_ePhArr) bcast(d, nBands, mpiUtil, root);
	}
	//e-ph sum rule if needed
	if(inEphLoop)
		bcast(state.dHePhSum, nBands*nBands, 3, mpiUtil, root);
}

void FeynWann::bcastState_inEphLoop(FeynWann::StateE& state, MPIUtil* mpiUtil, int root)
{
	if (mpiUtil->nProcesses() == 1) return; //no communictaion needed
	mpiUtil->bcast(state.ik, root);
	mpiUtil->bcast(&state.k[0], 3, root);
	//Energy and eigenvectors:
	bcast(state.E, nBands, mpiUtil, root);
	mpiUtil->bcast(state.withinRange, root);
	if (not state.withinRange) return; //Remaining quantities will never be used
	bcast(state.U, nBands, nBands, mpiUtil, root);
	//Velocity matrix, if needed:
	if (fwp.needVelocity)
	{
		for (int iDir = 0; iDir<3; iDir++)
			bcast(state.v[iDir], nBands, nBands, mpiUtil, root);
		state.vVec.resize(nBands);
		mpiUtil->bcastData(state.vVec, root);
	}
	//Spin matrix, if needed:
	if (fwp.needSpin)
	{
		for (int iDir = 0; iDir<3; iDir++)
			bcast(state.S[iDir], nBands, nBands, mpiUtil, root);
		state.Svec.resize(nBands);
		mpiUtil->bcastData(state.Svec, root);
	}
	//Linewidths, if needed:
	if (fwp.needLinewidth_ee) bcast(state.ImSigma_ee, nBands, mpiUtil, root);
	if (fwp.needLinewidth_ePh)
	{
		state.logImSigma_ePhArr.resize(FeynWannParams::fGrid_ePh.size());
		for (diagMatrix& d : state.logImSigma_ePhArr) bcast(d, nBands, mpiUtil, root);
	}
	if (fwp.needLinewidthP_ePh)
	{
		state.logImSigmaP_ePhArr.resize(FeynWannParams::fGrid_ePh.size());
		for (diagMatrix& d : state.logImSigmaP_ePhArr) bcast(d, nBands, mpiUtil, root);
	}
	//e-ph sum rule if needed
	bcast(state.dHePhSum, nBands*nBands, 3, mpiUtil, root);
}


void FeynWann::setState(FeynWann::StatePh& state)
{	assert(fwp.needPhonons);
	//Get force matrix:
	matrix Osqq = getMatrix(OsqW->getResult(state.iq), nModes, nModes);

	//Add polar corrections (LO-TO  splits) if any:
	if(polar)
	{	//Prefactor including denominator:
		int prodSup = OsqW->nkTot;
		matrix3<> G = (2.*M_PI)*inv(R);
		matrix3<> GT = ~G;
		//wrap q to BZ before qCart
		vector3<> qBZ = state.q;
		for(int iDir=0; iDir<3; iDir++)
			qBZ[iDir] -= floor(qBZ[iDir] + 0.5);
		vector3<> qCart = GT * qBZ;
		double prefac;
		if (truncDir < 3)
			prefac = (2.*M_PI) / (prodSup * omegaEff * qCart.length()) * lrs2D->wkernel(qCart);
		else
			prefac = (4.*M_PI) / (prodSup * Omega * epsInf.metric_length_squared(qCart));
		//Construct q.Z for each mode:
		diagMatrix qdotZbySqrtM(nModes);
		for(int iMode=0; iMode<nModes; iMode++)
			qdotZbySqrtM[iMode] = dot(Zeff[iMode], qCart) * invsqrtM[iMode];
		//Fourier transform cell weights to present q:
		matrix phase = zeroes(phononCellMap.size(), 1);
		complex* phaseData = phase.data();
		for(size_t iCell=0; iCell<phononCellMap.size(); iCell++)
			phaseData[iCell] = cis(2*M_PI*dot(state.q, phononCellMap[iCell]));
		matrix wTilde = phononCellWeights * phase; //nAtoms*nAtoms x 1 matrix
		wTilde.reshape(nAtoms, nAtoms);
		//Add corrections:
		complex* OsqqData = Osqq.data(); //iterating over nModes x nModes matrix
		int iMode2 = 0;
		for(int atom2=0; atom2<nAtoms; atom2++)
		for(int iDir2=0; iDir2<3; iDir2++)
		{	int iMode1 = 0;
			for(int atom1=0; atom1<nAtoms; atom1++)
			for(int iDir1=0; iDir1<3; iDir1++)
			{	*(OsqqData++) += prefac
					* wTilde(atom1,atom2) //cell weights
					* qdotZbySqrtM[iMode1] //charge and mass factor for mode 1
					* qdotZbySqrtM[iMode2]; //charge and mass factor for mode 2
				iMode1++;
			}
			iMode2++;
		}
	}
	
	//Diagonalize force matrix:
	Osqq.diagonalize(state.U, state.omega);
	for(double& omega: state.omega) omega = sqrt(std::max(0.,omega)); //convert to phonon frequency; discard imaginary
}

void FeynWann::bcastState(FeynWann::StatePh& state, MPIUtil* mpiUtil, int root)
{	if(mpiUtil->nProcesses()==1) return; //no communictaion needed
	mpiUtil->bcast(state.iq, root);
	mpiUtil->bcast(state.iqFine, root);
	mpiUtil->bcast(&state.q[0], 3, root);
	bcast(state.omega, nModes, mpiUtil, root);
	bcast(state.U, nModes, nModes, mpiUtil, root);
}


void FeynWann::setMatrix(const FeynWann::StateE& e1, const FeynWann::StateE& e2, const FeynWann::StatePh& ph, int ikPair, FeynWann::MatrixEph& m)
{	static StopWatch watch("FeynWann::setMatrix"); watch.start();
	m.e1 = &e1;
	m.e2 = &e2;
	m.ph = &ph;
	//Get the matrix elements for all modes together:
	matrix Mall = getMatrix(HePhW->getResult(ikPair), nBands*nBands, nModes);
	//Add long range polar corrections if required:
	if(polar)
	{	complex gLij;
		for(int iMode=0; iMode<nModes; iMode++) //in Cartesian atom displacement basis
		{	if (truncDir < 3)
				gLij =  complex(0,1)
					* ((2*M_PI) * invsqrtM[iMode] / (omegaEff))
					*  (*lrs2D)(ph.q, Zeff[iMode], atpos[iMode/3]);
			else
				gLij =  complex(0,1)
					* ((4*M_PI) * invsqrtM[iMode] / (Omega))
					*  (*lrs)(ph.q, Zeff[iMode], atpos[iMode/3]);
			for(int b=0;  b<nBands; b++)
				Mall.data()[Mall.index(b*(nBands+1), iMode)] += gLij; //diagonal only
		}
	}
	//Apply sum rule correction:
	complex* Mdata = Mall.dataPref();
	for(int iAtom=0; iAtom<nAtoms; iAtom++)
	{	int nData = m.e1->dHePhSum.nData();
		double alpha = (-0.5/nAtoms)*invsqrtM[3*iAtom];
		eblas_zaxpy(nData, alpha, m.e1->dHePhSum.dataPref(),1, Mdata,1);
		eblas_zaxpy(nData, alpha, m.e2->dHePhSum.dataPref(),1, Mdata,1);
		Mdata += nData;
	}
	//Apply phonon transformation:
	Mall = Mall * m.ph->U; //to phonon eigenbasis
	//Extract matrices for each phonon mode:
	const double omegaPhCut = 1e-6;
	m.M.resize(nModes);
	for(int iMode=0; iMode<nModes; iMode++)
		m.M[iMode] = sqrt(m.ph->omega[iMode]<omegaPhCut ? 0. : 0.5/m.ph->omega[iMode]) //frequency-dependent phonon amplitude
			* (dagger(m.e1->U) * getMatrix(Mall.data(), nBands, nBands, iMode) * m.e2->U); //to E1 and E2 eigenbasis
	watch.stop();
}


//----------- class FeynWann::StateE -------------
inline double interpQuartic(const std::vector<diagMatrix>& Y, int n, double f)
{	//Get bernstein coeffs
	double a0 = Y[0][n];
	double a4 = Y[4][n];
	double a1 = (1./12)*(-13.*Y[0][n]+48.*Y[1][n]-36.*Y[2][n]+16.*Y[3][n]-3.*Y[4][n]);
	double a3 = (1./12)*(-13.*Y[4][n]+48.*Y[3][n]-36.*Y[2][n]+16.*Y[1][n]-3.*Y[0][n]);
	double a2 = (1./18)*(13.*(Y[0][n]+Y[4][n])-64.*(Y[1][n]+Y[3][n])+120.*Y[2][n]);
	//Evaluate bernstein polynomial
	//--- 1
	double b0 = a0+f*(a1-a0);
	double b1 = a1+f*(a2-a1);
	double b2 = a2+f*(a3-a2);
	double b3 = a3+f*(a4-a3);
	//--- 2
	double c0 = b0+f*(b1-b0);
	double c1 = b1+f*(b2-b1);
	double c2 = b2+f*(b3-b2);
	//--- 3
	double d0 = c0+f*(c1-c0);
	double d1 = c1+f*(c2-c1);
	//--- 4
	return d0+f*(d1-d0);
}
double FeynWann::StateE::ImSigma_ePh(int n, double f) const
{	return exp(interpQuartic(logImSigma_ePhArr, n, f));
}
double FeynWann::StateE::ImSigmaP_ePh(int n, double f) const
{	return exp(interpQuartic(logImSigmaP_ePhArr, n, f));
}

//Elementwise std::pow of a matrix
template<typename PowType> matrix3<> powElemWise(const matrix3<>& m, PowType n)
{
	matrix3<> result;
	for (int i = 0; i<3; i++)
	for (int j = 0; j<3; j++)
		result(i, j) = std::pow(m(i, j), n);
	return result;
}

//Report a tensor with error estimates
void reportResult(const std::vector<matrix3<>>& result, string resultName, double unit, string unitName, FILE* fp, bool invAvg)
{
	matrix3<> sum, sumSq; int N = 0;
	for (size_t block = 0; block<result.size(); block++)
	{
		N++;
		matrix3<> term = invAvg ? inv(result[block]) : result[block];
		sum += term;
		sumSq += powElemWise(term, 2);
	}
	matrix3<> resultMean = (1. / N)*sum;
	matrix3<> resultStd = powElemWise((1. / N)*sumSq - powElemWise(resultMean, 2), 0.5); //element-wise std. deviation
	if (invAvg)
	{
		resultMean = inv(resultMean); //harmonic matrix mean (inverse of mean matrix inverse)
		resultStd = resultMean * resultStd * resultMean; //propagate error in reciprocal
	}
	//Print result:
	for (int i = 0; i<3; i++)
	{
		char mOpen[] = "/|\\", mClose[] = "\\|/";
		fprintf(fp, "%20s%c", i == 1 ? (resultName + " = ").c_str() : "", mOpen[i]);
		for (int j = 0; j<3; j++) fprintf(fp, " %12lg", resultMean(i, j) / unit);
		if (N>1)
		{
			fprintf(fp, " %c%5s%c", mClose[i], i == 1 ? " +/- " : "", mOpen[i]);
			for (int j = 0; j<3; j++) fprintf(fp, " %12lg", fabs(resultStd(i, j)) / unit);
			fprintf(fp, " %c %s\n", mClose[i], i == 1 ? unitName.c_str() : "");
		}
		else fprintf(fp, " %c %s\n", mClose[i], i == 1 ? unitName.c_str() : "");
	}
	fprintf(fp, "\n");
}

//Report a scalar with error estimates:
void reportResult(const std::vector<double>& result, string resultName, double unit, string unitName, FILE* fp, bool invAvg)
{
	double sum = 0., sumSq = 0.; int N = 0;
	for (size_t block = 0; block<result.size(); block++)
	{
		N++;
		double term = invAvg ? 1. / result[block] : result[block];
		sum += term;
		sumSq += term*term;
	}
	double resultMean = sum / N;
	double resultStd = sqrt(sumSq / N - std::pow(resultMean, 2));
	if (invAvg)
	{
		resultMean = 1. / resultMean; //harmonic mean
		resultStd *= std::pow(resultMean, 2); //propagate error in reciprocal
	}
	if (N>1)
		fprintf(fp, "%17s = %12lg +/- %12lg %s\n", resultName.c_str(), resultMean / unit, fabs(resultStd) / unit, unitName.c_str());
	else
		fprintf(fp, "%17s = %12lg %s\n", resultName.c_str(), resultMean / unit, unitName.c_str());
}