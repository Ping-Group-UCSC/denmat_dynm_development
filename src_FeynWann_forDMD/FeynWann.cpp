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
#include "FeynWann_internal.h"
#include "InputMap.h"
#include <core/BlasExtra.h>
#include <core/Random.h>
#include <wannier/WannierMinimizer.h>
#include <fftw3-mpi.h>
#include "config.h"

#ifdef TDEP_ENABLED
#include "tdep_wrapper.h"
#endif

FeynWannParams::FeynWannParams(InputMap* inputMap)
: iSpin(0), totalEprefix("Wannier/totalE"), phononPrefix("Wannier/phonon"), wannierPrefix("Wannier/wannier"),
needSymmetries(false), needPhonons(false), needVelocity(false), needSpin(false), needL(false), needQ(false),
needLinewidth_ee(false), needLinewidth_ePh(false), needLinewidthP_ePh(false),
ePhHeadOnly(false), maskOptimize(false), bandSumLQ(false),
orbitalZeeman(false), EzExt(0.), scissor(0.), EshiftWeight(0.), enforceKramerDeg(0.), degeneracyThreshold(1E-4*eV), tdep(false),
Bext(vector3<>(0., 0., 0.)), needLayer(false), trunc_iDi_outer(false),
Bin_model("none"), alpha_Bin(0), kcut_model(1), kcut_width_model(0.1), dir_2drb(vector3<>(0, 0, 0)), pshk(vector3<>(0, 0, 0)), pshs(vector3<>(0, 0, 0)),
bStart_model(0), bEnd_model(-1), renorm_Bin(0),
scissor_above_band(0), band_scissor(0)
{
	if(inputMap)
	{	const double nm = 10*Angstrom;
		Bext = inputMap->getVector("Bext", vector3<>(0.,0.,0.)) * Tesla;
		orbitalZeeman = inputMap->getBool("orbitalZeeman", false);
		if(Bext.length_squared())
		{	needSpin = true; //will be disabled by FeynWann for non-SOC mode
			needL = orbitalZeeman;
			logPrintf("Adding spin%s to requirements for non-zero Bext.\n", needL ? " and L" : "");
		}
		EzExt = inputMap->get("EzExt", 0.) * eV/nm;
		scissor = inputMap->get("scissor", 0.) * eV;
		EshiftWeight = inputMap->get("EshiftWeight", 0.) * eV;
		bandSumLQ = inputMap->getBool("bandSumLQ", false);
		enforceKramerDeg = inputMap->getBool("enforceKramerDeg", false);
		degeneracyThreshold = inputMap->get("degeneracyThreshold", 1E-4) * eV;
		tdep = inputMap->getBool("tdep", false);
		#ifndef TDEP_ENABLED
		if(tdep) die("Must link with TDEP code to use tdep.\n");
		#endif
		//JX
		trunc_iDi_outer = inputMap->get("trunc_iDi_outer", 0);
		Bin_model = inputMap->getString("Bin_model", "none");
		kcut_model = inputMap->get("kcut_model", 1); assert(kcut_model > 0);
		kcut_width_model = inputMap->get("kcut_width_model", 0.1); assert(kcut_width_model > 0);
		bool allowed = Bin_model == "none" || Bin_model == "read"|| Bin_model == "3d-rashba" || Bin_model == "3d-rb" || Bin_model == "2d-rashba" || Bin_model == "2d-rb" || Bin_model == "psh";
		if (not allowed) logPrintf("Bin_model should be none, read, 3d-rb, 3d-rashba, 2d-rb, 2d-rashba, psh");
		assert(allowed);
		alpha_Bin = inputMap->get("alpha_Bin", 0) * eV * Angstrom;
		dir_2drb = normalize(inputMap->getVector("dir_2drb", vector3<>(0, 0, 1)));
		pshk = normalize(inputMap->getVector("pshk", vector3<>(1, 0, 0)));
		pshs = normalize(inputMap->getVector("pshs", vector3<>(0, 0, 1)));
		bStart_model = inputMap->get("bStart_model", 0);
		bEnd_model = inputMap->get("bEnd_model", -1);
		renorm_Bin = inputMap->get("renorm_Bin", 0);
		scissor_above_band = inputMap->get("scissor_above_band", 0.) * eV;
		band_scissor = inputMap->get("band_scissor", 0);
		//JX
	}
}


void FeynWannParams::printParams() const
{	logPrintf("Bext = "); Bext.print(globalLog, " %lg ");
	logPrintf("orbitalZeeman = %s\n", orbitalZeeman ? "yes" : "no");
	logPrintf("EzExt = %lg\n", EzExt);
	logPrintf("scissor = %lg\n", scissor);
	logPrintf("EshiftWeight = %lg\n", EshiftWeight);
	logPrintf("bandSumLQ = %s\n", bandSumLQ ? "yes" : "no");
	logPrintf("enforceKramerDeg = %s\n", enforceKramerDeg ? "yes" : "no");
	logPrintf("degeneracyThreshold = %lg\n", degeneracyThreshold);
	logPrintf("tdep = %s\n", tdep ? "yes" : "no");
	//JX
	logPrintf("trunc_iDi_outer = %d\n", trunc_iDi_outer);
	logPrintf("Bin_model = %s\n", Bin_model.c_str());
	logPrintf("alpha_Bin = %lg\n", alpha_Bin);
	logPrintf("kcut_model = %lg\n", kcut_model);
	logPrintf("kcut_width_model = %lg\n", kcut_width_model);
	logPrintf("dir_2drb = %lg %lg %lg\n", dir_2drb[0], dir_2drb[1], dir_2drb[2]);
	logPrintf("pshk = %lg %lg %lg\n", pshk[0], pshk[1], pshk[2]);
	logPrintf("pshs = %lg %lg %lg\n", pshs[0], pshs[1], pshs[2]);
	logPrintf("bStart_model = %d\n", bStart_model);
	logPrintf("bEnd_model = %d\n", bEnd_model);
	logPrintf("bEnd_model = %d\n", bEnd_model);
	logPrintf("renorm_Bin = %d\n", renorm_Bin);
	logPrintf("scissor_above_band = %lg\n", scissor_above_band);
	logPrintf("band_scissor = %d\n", band_scissor);
	//JX
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
	ip.packageName = "JDFTx";
	ip.versionString = "1.7";
	ip.versionHash = "";
	ip.description = description;
	initSystemCmdline(argc, argv, ip);
	fftw_mpi_init();
	return ip;
}


void FeynWann::finalize()
{	fftw_mpi_cleanup();
	finalizeSystem();
}


FeynWann::FeynWann(FeynWannParams& fwp)
: fwp(fwp), nAtoms(0), nSpins(0), nSpinor(0), spinWeight(0), mu(NAN), nElectrons(0),
EminInner(-INFINITY), EmaxInner(+INFINITY), polar(false), ePhEstart(0.), ePhEstop(-1.),
tTransformByCompute(1), tTransformByComputeD(1), inEphLoop(false),
isMetal(false), energyOnly(false), 
Bso(vector3<>(0,0,0)), gfac(matrix3<>(gElectron, gElectron, gElectron))//JX
{
	//Create inter-group communicator if requested:
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
	nBandsDFT = 0; //JX //number of DFT bands (>= this->nBands = # Wannier bands)
	nStatesDFT = 0; //JX //number of reduced k-pts * spins in DFT
	while (!ifs.eof())
	{	string line; getline(ifs, line);
		if(line.find("Initializing the grid") != string::npos)
		{	getline(ifs, line); //skip the line containing "R = "
			for(int j=0; j<3; j++)
			{	getline(ifs, line);
				sscanf(line.c_str(), "[ %lf %lf %lf ]", &R(j,0), &R(j,1), &R(j,2));
			}
			Omega = fabs(det(R));
			G = 2 * M_PI * inv(R); GT = ~G; GGT = G * (~G); //JX
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
		isMetal = false;//JX
		int nValence = int(round(nElectrons/(nSpins*spinWeight))); //number of valence bands
		if(fabs(nValence*nSpins*spinWeight-nElectrons) > 1e-6)
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
	else isMetal = true;//JX
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
	
	//Check for Wannier inner window:
	fname = fwp.wannierPrefix + ".out";
	logPrintf("\nReading '%s' ... ", fname.c_str()); logFlush();
	ifs.open(fname); if (!ifs.is_open()) die("could not open file.\n");
	while(!ifs.eof())
	{	string line; getline(ifs, line);
		if(line.find("wannier  \\") != string::npos)
		{	//At start of wannier command print
			while(!ifs.eof())
			{	getline(ifs, line);
				if(line[0] != '\t') break; //no longer within wannier command
				//Parse key and values within:
				istringstream iss(line);
				string key;
				iss >> key;
				if(key == "innerWindow")
				{	iss >> EminInner >> EmaxInner;
					if(!iss.good()) die("Failed to parse innerWindow.\n");
				}
			}
		}
	}
	ifs.close();
	logPrintf("done.\n");
	logPrintf("EminInner = %lg\nEmaxInner = %lg\n", EminInner, EmaxInner);
	EminInner -= mu; //since energies will always be referenced against mu
	EmaxInner -= mu; 

	//Read cell weights:
	cellWeights = readCellWeights(fwp.wannierPrefix + ".mlwfCellWeights" + spinSuffix, cellMap.size(), nBands, nBands);
	
	//Initialize phonon properties:
	realPartOnly = (nSpinor==1);
	offsetDim = kfold; //size of an offset is determined by electronic k-points by default
	kfoldSup = vector3<int>(1,1,1); //no additional k-point sampling needed beyond offsetDim
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
		offsetDim = phononSup; //size of an offset is limited by phonon supercell
		
		//Read phonon basis:
		invsqrtM = readPhononBasis(fwp.totalEprefix + ".phononBasis");//confusing! read 1/sqrt(mass) in reality
		phononBasis = readPhononBasisFromFile(fwp.totalEprefix + ".phononBasis");//JX

		if(fwp.tdep)
		{
			#ifdef TDEP_ENABLED
			logPrintf("\n---------- TDEP Initialization ----------\n");
			int verbosity = mpiWorld->isHead() ? 2 : -10;
			tdep_initialize_(&verbosity);
			logPrintf("\n");
			#endif
		}
		{	//Read phonon cell map:
			fname = fwp.totalEprefix + ".phononCellMap";
			if(fileSize((fname + "Corr").c_str()) > 0) //corrected force matrix cell map exists
				fname += "Corr";
			phononCellMap = readCellMap(fname);
			
			//Read phonon force matrix
			fname = fwp.totalEprefix + ".phononOmegaSq";
			if(fileSize((fname + "Corr").c_str()) > 0) //corrected force matrix exists
				fname += "Corr";
			OsqW = std::make_shared<DistributedMatrix>(fname, true, //phonon omegaSq is always real
				mpiGroup, nModes*nModes, phononCellMap, offsetDim, false, mpiInterGroup);
		}
		
		//Read cell maps for electron-phonon matrix elements and sum rule:
		ePhCellMap = readCellMap(fwp.wannierPrefix + ".mlwfCellMapPh" + spinSuffix);
		ePhCellMapSum = readCellMap(fwp.wannierPrefix + ".mlwfCellMapPhSum" + spinSuffix);

		//Read e-ph cell weights (atom x band weights for each cell in ePhCellMap):
		fname = fwp.wannierPrefix + ".mlwfCellWeightsPh" + spinSuffix;
		std::vector<matrix> ePhCellWeights = readCellWeights(fname, ePhCellMap.size(), nAtoms, nBands);
		
		//Read electron-phonon matrix elements
		fname = fwp.wannierPrefix + ".mlwfHePh" + spinSuffix;
		HePhW = std::make_shared<DistributedMatrix>(fname, realPartOnly,
			mpiGroup, nModes*nBands*nBands, ePhCellMap, offsetDim, true, mpiInterGroup, &ePhCellWeights);
		
		//Read electron-phonon matrix element sum rule
		fname = fwp.wannierPrefix + ".mlwfHePhSum" + spinSuffix;
		HePhSumW = std::make_shared<DistributedMatrix>(fname, realPartOnly,
			mpiGroup, 3*nBands*nBands, ePhCellMapSum, offsetDim, false, mpiInterGroup);
		
		//Read gradient matrix element for e-ph sum rule
		Dw = readE("mlwfD", 3);
		
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
		
		//Benchmark e-ph transform and compute to optimize masked computations, if needed:
		if(fwp.maskOptimize)
		{	logPrintf("Benchmarking e-ph transform and single-point compute: "); logFlush();
			const int nRepeat = 5;
			#define TIMErepeated(funcName) \
				double funcName##Time = 0.; \
				{	double tStart = clock_sec(); \
					for(int iRepeat=0; iRepeat < nRepeat; iRepeat++) \
					{	HePhW->funcName(vector3<>(), vector3<>()); \
					} \
					funcName##Time = (clock_sec() - tStart) / nRepeat; \
				}
			TIMErepeated(compute)
			TIMErepeated(transform)
			#undef TIMErepeated
			tTransformByCompute = int(floor(transformTime/computeTime));
			logPrintf("tCompute[s]: %lg tTransform[s]: %lg\n", computeTime, transformTime);
			logPrintf("Will switch from transform to compute when mask count <= %d\n", tTransformByCompute);
		}
	}
		
	//Initialize defect properties:
	if(fwp.needDefect.length())
	{	logPrintf("\nInitializing defect '%s':\n", fwp.needDefect.c_str());
		
		//Get defect supercell from wannier output file:
		fname = fwp.wannierPrefix + ".out";
		logPrintf("Reading '%s' ... ", fname.c_str()); logFlush();
		ifs.open(fname); if(!ifs.is_open()) die("could not open file.\n");
		while(!ifs.eof())
		{	string line; getline(ifs, line);
			if(line.find("defect-supercell") == 0)
			{	//at start of wannier command print
				string key, name;
				istringstream iss(line);
				iss >> key >> name;
				if(name == fwp.needDefect)
				{	iss >> defectSup[0] >> defectSup[1] >> defectSup[2];
					break;
				}
			}
		}
		ifs.close();
		if(defectSup.length_squared())
		{	logPrintf("done. Defect supercell: ");
			defectSup.print(globalLog, " %d ");
		}
		else die("could not determine defect supercell.\n\n");
		for(int iDir=0; iDir<3; iDir++)
		{	kfoldSup[iDir] = kfold[iDir] / defectSup[iDir];
			if(kfoldSup[iDir] * defectSup[iDir] != kfold[iDir])
				die("kfold is not a multiple of defect supercell.\n");
		}
		if(fwp.needPhonons and (not (phononSup == defectSup)))
			die("Phonon and defect supercells don't match; this is currently needed to calculate both in same run.\n");
		offsetDim = defectSup; //size of an offset is limited by defect supercell
		
		//Read corresponding cell map and weights:
		defectCellMap = readCellMap(fwp.wannierPrefix + ".mlwfCellMapD_" + fwp.needDefect + spinSuffix);
		std::vector<matrix> defectCellWeights = readCellWeights(
			fwp.wannierPrefix + ".mlwfCellWeightsD_" + fwp.needDefect + spinSuffix,
			defectCellMap.size(), 1, nBands);
		
		//Read electron-defect matrix elements:
		fname = fwp.wannierPrefix + ".mlwfHD_" + fwp.needDefect + spinSuffix;
		HdefectW = std::make_shared<DistributedMatrix>(fname, realPartOnly,
			mpiGroup, nBands*nBands, defectCellMap, offsetDim, true, mpiInterGroup, &defectCellWeights);
		
		//Benchmark e-defect transform and compute to optimize masked computations, if needed:
		if(fwp.maskOptimize)
		{	logPrintf("Benchmarking e-defect transform and single-point compute: "); logFlush();
			const double tMin = 0.5; //time for at least 0.5 s
			const int nMin = 3; //time at least 3 evaluations
			#define TIMErepeated(funcName) \
				double funcName##Time = 0.; \
				{	double tStart = clock_sec(), t=0.; \
					int nTries = 0; \
					while(nTries<nMin or t<tMin) \
					{	HdefectW->funcName(vector3<>(), vector3<>()); \
						nTries++; \
						t = clock_sec()-tStart; \
					} \
					funcName##Time = t / nTries; \
				}
			TIMErepeated(compute)
			TIMErepeated(transform)
			#undef TIMErepeated
			tTransformByComputeD = int(floor(transformTime/computeTime));
			logPrintf("tCompute[s]: %lg tTransform[s]: %lg\n", computeTime, transformTime);
			logPrintf("Will switch from transform to compute when mask count <= %d\n", tTransformByComputeD);
		}
	}
	
	//Read wannier hamiltonian
	fname = fwp.wannierPrefix + ".mlwfH" + spinSuffix;
	Hw = readE("mlwfH");
	//--- optional offset by slab weights:
	if(fwp.EshiftWeight)
	{	std::shared_ptr<DistributedMatrix> Ww = readE("mlwfW");
		axpy(fwp.EshiftWeight, Ww->mat, Hw->mat);
	}
	
	//Velocity matrix elements
	if(fwp.needVelocity or (fwp.needL or fwp.needQ)) //also needed for long-raneg correction of R*P
		Pw = readE("mlwfP", 3);
	//Spin matrix elements
	if(not isRelativistic()) fwp.needSpin = false; //spin only available in relatvistic mode
	if(fwp.needSpin) Sw = readE("mlwfS", 3);
	//R*P matrix elements for angular momentum and/or electric quadrupole
	if(fwp.needRP())
	{	RPw = readE("mlwfRP", 9);
		//Setup dH/dk for long-range correction:
		for(int iDir=0; iDir<3; iDir++)
			HprimeW[iDir] = std::make_shared<DistributedMatrix>("", realPartOnly, //data taken from Hw
				mpiGroup, nBands*nBands, cellMap, offsetDim, false, mpiInterGroup, &cellWeights, &kfold,
				iDir, Hw.get(), &R);
	}
	//z position matrix elements
	if(fwp.EzExt) Zw = readE("mlwfZ");
	if (fwp.needLayer) Layerw = readE("mlwfW");//JX
	//Linewidths:
	if(fwp.needLinewidth_ee) ImSigma_eeW = readE("mlwfImSigma_ee");
	if(fwp.needLinewidth_ePh) ImSigma_ePhW = readE("mlwfImSigma_ePh", FeynWannParams::fGrid_ePh.size());
	if(fwp.needLinewidthP_ePh) ImSigmaP_ePhW = readE("mlwfImSigmaP_ePh", FeynWannParams::fGrid_ePh.size());
	if(fwp.needLinewidth_D.length()) ImSigma_DW = readE("mlwfImSigma_D_" + fwp.needLinewidth_D);
	if(fwp.needLinewidthP_D.length()) ImSigmaP_DW = readE("mlwfImSigmaP_D_" + fwp.needLinewidth_D);
	logPrintf("\n");
	
	//Initialize q-mesh offsets that will cover k-mesh:
	qOffset.clear();
	vector3<int> iqOffset;
	matrix3<> kfoldInv = inv(Diag(vector3<>(kfold)));
	for(iqOffset[0]=0; iqOffset[0]<kfoldSup[0]; iqOffset[0]++)
	for(iqOffset[1]=0; iqOffset[1]<kfoldSup[1]; iqOffset[1]++)
	for(iqOffset[2]=0; iqOffset[2]<kfoldSup[2]; iqOffset[2]++)
		qOffset.push_back(kfoldInv * iqOffset);
}


std::shared_ptr<DistributedMatrix> FeynWann::readE(string varname, int nVars) const
{	return std::make_shared<DistributedMatrix>(fwp.wannierPrefix + "." + varname + spinSuffix, realPartOnly,
			mpiGroup, nBands*nBands*nVars, cellMap, offsetDim, false, mpiInterGroup, &cellWeights, &kfold);
}


void FeynWann::free()
{	Hw = 0;
	Pw = 0;
	Sw = 0;
	RPw = 0;
	Zw = 0;
	Layerw = 0;//JX
	for(int iDir=0; iDir<3; iDir++) HprimeW[iDir] = 0;
	ImSigma_eeW = 0;
	ImSigma_ePhW = 0;
	ImSigmaP_ePhW = 0;
	ImSigma_DW = 0;
	ImSigmaP_DW = 0;
	OsqW = 0;
	HePhW = 0;
	HePhSumW = 0;
	Dw = 0;
	HdefectW = 0;
}
