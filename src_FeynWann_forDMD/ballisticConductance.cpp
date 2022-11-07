/*-------------------------------------------------------------------
Copyright 2020 Ravishankar Sundararaman

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
#include <core/Units.h>
#include <core/LatticeUtils.h>

struct CollectBallisticConductance
{	const FeynWann& fw;
	const vector3<int> NkFine;
	const double dmu;
	const double invSmearWidth;
	std::vector<int> invertList;
	std::vector<matrix3<>> symCart; //Cartesian symmetry matrices
	const double prefacZZ; //prefactor to kx and ky resolved ballistic conductance
	const double prefac; //prefactor to total ballistic conductance
	double wOffsetCur;
	matrix3<> G; //net ballistic conductance
	vector3<> GL; //net Landauer conductance
	std::vector<double> Gzz, GLz; //G_zz and GL_z at specific kx and ky
	std::vector<double> vNum, vDen; //numerator and denominator for average |v| at each kx and ky
	
	CollectBallisticConductance(const FeynWann& fw, vector3<int> NkFine, double dmu, double smearWidth, std::vector<int> invertList)
	: fw(fw), NkFine(NkFine), dmu(dmu), invSmearWidth(1./smearWidth), invertList(invertList),
		prefacZZ(fw.spinWeight/(fw.Omega*fw.sym.size()*invertList.size()*NkFine[2])),
		prefac(prefacZZ/(NkFine[0]*NkFine[1])),
		Gzz(NkFine[0]*NkFine[1]), GLz(NkFine[0]*NkFine[1]),
		vNum(NkFine[0]*NkFine[1]), vDen(NkFine[0]*NkFine[1])
	{
		for(const SpaceGroupOp& op: fw.sym)
			symCart.push_back(fw.R * op.rot * inv(fw.R));
	}
	
	//Convert component of k to corresponding k-mesh coordinate:
	inline int meshCoord(vector3<> k, int iDir) const
	{	return positiveRemainder(int(round(k[iDir]*NkFine[iDir])), NkFine[iDir]);
	}
	
	void eProcess(const FeynWann::StateE& state)
	{	//Determine mapping of k under symmetries:
		std::vector<int> kIndex; kIndex.reserve(symCart.size()*invertList.size());
		for(const SpaceGroupOp& op: fw.sym)
			for(int invert: invertList)
			{	vector3<> k = invert * state.k * op.rot;
				kIndex.push_back(meshCoord(k,0) * NkFine[1] + meshCoord(k,1));
			}
		//Collect contributions:
		for(int b=0; b<fw.nBands; b++)
		{	double Ediff = invSmearWidth*(state.E[b] - dmu);
			if(fabs(Ediff)<35.) //fermi prime non-zero at double precision
			{	const vector3<>& vCur = state.vVec[b];
				double mfPrime = 0.25*invSmearWidth * std::pow(cosh(0.5*Ediff), -2); //-df/dE
				double vMag = vCur.length();
				double weight = wOffsetCur * mfPrime / std::max(1e-6, vMag);
				double weightL = wOffsetCur * mfPrime * 0.5; //weight for Landauer version
				int* kIndexPtr = kIndex.data();
				for(size_t iSym=0; iSym<symCart.size(); iSym++)
					for(int invert: invertList)
					{	vector3<> v = symCart[iSym] * vCur * invert;
						vector3<> vAbs(fabs(v[0]), fabs(v[1]), fabs(v[2]));
						G += prefac * weight * outer(v,v);
						GL += prefac * weightL * vAbs;
						Gzz[*kIndexPtr] += prefacZZ * weight * v[2]*v[2];
						GLz[*kIndexPtr] += prefacZZ * weightL * vAbs[2];
						vNum[*kIndexPtr] += prefacZZ * weightL * vMag;
						vDen[*kIndexPtr] += prefacZZ * weightL;
						kIndexPtr++;
					}
			}
		}
	}
	static void eProcess(const FeynWann::StateE& state, void* params)
	{	((CollectBallisticConductance*)params)->eProcess(state);
	}
};

inline void writeProfile(const std::vector<double>& G, double scale, vector3<int> Nk, string fname)
{	FILE* fp = fopen(fname.c_str(), "w");
	const double* Gdata = G.data();
	for(int ik0=0; ik0<Nk[0]; ik0++)
	{	for(int ik1=0; ik1<Nk[1]; ik1++) fprintf(fp, "%lg ", *(Gdata++) * scale);
		fprintf(fp, "\n");
	}
	fclose(fp);
}

int main(int argc, char** argv)
{   InitParams ip =  FeynWann::initialize(argc, argv, "Ballistic and Landauer conductance calculator.");

	//Read input file:
	InputMap inputMap(ip.inputFilename);
	const double dmu = inputMap.get("dmu", 0.) * eV; //optional shift in chemical potential from neutral value
	const double smearWidth = inputMap.get("smearWidth") * eV;
	const int iSpin = inputMap.get("iSpin", 0); //spin channel (default 0)
	const int NkMultAll = int(round(inputMap.get("NkMult"))); //increase in number of k-points for electron k-mesh
	FeynWannParams fwp(&inputMap);
	
	vector3<int> NkMult;
	NkMult[0] = inputMap.get("NkxMult", NkMultAll); //override increase in x direction
	NkMult[1] = inputMap.get("NkyMult", NkMultAll); //override increase in y direction
	NkMult[2] = inputMap.get("NkzMult", NkMultAll); //override increase in z direction
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("dmu = %lg\n", dmu);
	logPrintf("smearWidth = %lg\n", smearWidth);
	logPrintf("iSpin = %d\n", iSpin);
	logPrintf("NkMult = "); NkMult.print(globalLog, " %d ");
	fwp.printParams();
	
	//Initialize FeynWann:
	fwp.iSpin = iSpin;
	fwp.needSymmetries = true;
	fwp.needVelocity = true;
	FeynWann fw(fwp);
	
	//Check NkMult compatibility with symmetries:
	for(const SpaceGroupOp& op: fw.sym)
	{	//Similar to Symmetries::checkFFTbox in JDFTx
		matrix3<int> mMesh = Diag(NkMult) * op.rot;
		for(int i=0; i<3; i++)
			for(int j=0; j<3; j++)
				if(mMesh(i,j) % NkMult[j] == 0)
					mMesh(i,j) /= NkMult[j];
				else
				{	logPrintf("NkMult not commensurate with symmetry matrix:\n");
					op.rot.print(globalLog, " %2d ");
					op.a.print(globalLog, " %lg ");
					die("NkMult not commensurate with symmetries.\n");
				}
	}
	
	//Construct NkMult mesh:
	std::vector<vector3<>> kMult;
	vector3<int> NkFine;
	for(int iDir=0; iDir<3; iDir++)
	{	if(fw.isTruncated[iDir] && NkMult[iDir]!=1)
		{	logPrintf("Setting NkMult = 1 along truncated direction %d.\n", iDir+1);
			NkMult[iDir] = 1; //no multiplication in truncated directions
		}
		NkFine[iDir] = fw.kfold[iDir] * NkMult[iDir];
	}
	matrix3<> NkMultInv = inv(Diag(vector3<>(NkMult)));
	vector3<int> ikMult;
	for(ikMult[0]=0; ikMult[0]<NkMult[0]; ikMult[0]++)
	for(ikMult[1]=0; ikMult[1]<NkMult[1]; ikMult[1]++)
	for(ikMult[2]=0; ikMult[2]<NkMult[2]; ikMult[2]++)
		kMult.push_back(NkMultInv * ikMult);
	logPrintf("Effective interpolated k-mesh dimensions: ");
	NkFine.print(globalLog, " %d ");
	
	//Reduce under symmetries (simplified version of Symmetries::reduceKmesh from JDFTx):
	std::vector<vector3<>> k0; //array of k-mesh offsets
	std::vector<double> wk0; //corresponding weights
	//--- Compile list of inversions to check:
	std::vector<int> invertList;
	invertList.push_back(+1);
	invertList.push_back(-1);
	for(const SpaceGroupOp& op: fw.sym)
		if(op.rot==matrix3<int>(-1,-1,-1))
		{	invertList.resize(1); //inversion explicitly found in symmetry list, so remove from invertList
			break;
		}
	matrix3<> G = 2*M_PI*inv(fw.R), GGT = G*(~G);
	matrix3<> kfoldInv = inv(Diag(vector3<>(fw.kfold)));
	if(mpiWorld->isHead())
	{	//compile kpoint map:
		PeriodicLookup<vector3<>> plook(kMult, GGT);
		std::vector<bool> kDone(kMult.size(), false);
		for(size_t iSrc=0; iSrc<kMult.size(); iSrc++)
			if(!kDone[iSrc])
			{	double w = 0.; //weight of current point
				for(int invert: invertList)
					for(const SpaceGroupOp& op: fw.sym)
					{	size_t iDest = plook.find(invert * kMult[iSrc] * op.rot);
						if(iDest!=string::npos && (!kDone[iDest]))
						{	kDone[iDest] = true; //iDest in iSrc's orbit
							w += 1.; //increase weight of iSrc
						}
					}
				//add corresponding offset:
				k0.push_back(kfoldInv * kMult[iSrc]);
				wk0.push_back(w);
			}
	}
	//--- make available on all processes
	int nOffsets = k0.size(); mpiWorld->bcast(nOffsets);
	k0.resize(nOffsets); mpiWorld->bcastData(k0);
	wk0.resize(nOffsets); mpiWorld->bcastData(wk0);
	logPrintf("\n%lu offsets in NkMult mesh reduced to %d under symmetries.\n", kMult.size(), nOffsets);
	if(mpiWorld->isHead()) logPrintf("%d electron k-mesh offsets parallelized over %d process groups.\n", nOffsets, mpiGroupHead->nProcesses());
	
	logPrintf("\n");
	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		fw.free();
		FeynWann::finalize();
		return 0;
	}
	
	//Initialize sampling parameters:
	int oStart=0, oStop=0;
	if(mpiGroup->isHead())
		TaskDivision(nOffsets, mpiGroupHead).myRange(oStart, oStop);
	mpiGroup->bcast(oStart);
	mpiGroup->bcast(oStop);
	int noMine = oStop-oStart; //number of offsets handled by current group
	int oInterval = std::max(1, int(round(noMine/50.))); //interval for reporting progress
	
	//Collect conductance:
	logPrintf("\nCollecting ballistic conductance: "); logFlush();
	CollectBallisticConductance cbc(fw, NkFine, dmu, smearWidth, invertList);
	for(int o=oStart; o<oStop; o++)
	{	cbc.wOffsetCur = wk0[o];
		fw.eLoop(k0[o], CollectBallisticConductance::eProcess, &cbc);
		if((o-oStart+1)%oInterval==0) { logPrintf("%d%% ", int(round((o-oStart+1)*100./noMine))); logFlush(); }
	}
	logPrintf("done.\n"); logFlush();
	mpiWorld->allReduce(cbc.G, MPIUtil::ReduceSum);
	mpiWorld->allReduce(cbc.GL, MPIUtil::ReduceSum);
	mpiWorld->allReduceData(cbc.Gzz, MPIUtil::ReduceSum);
	mpiWorld->allReduceData(cbc.GLz, MPIUtil::ReduceSum);
	mpiWorld->allReduceData(cbc.vNum, MPIUtil::ReduceSum);
	mpiWorld->allReduceData(cbc.vDen, MPIUtil::ReduceSum);
	
	//Report ballistic resistance:
	double rhoLambdaUnit = 1E-16 * Ohm * pow(meter,2);
	matrix3<> rhoLambda = inv(cbc.G);
	logPrintf("\nrho*lambda [10^-16 Ohm-m^2]:\n");
	(rhoLambda*(1./rhoLambdaUnit)).print(globalLog, " %lg ", true, 1e-12);
	
	//Report landauer resistance:
	vector3<> Rlandauer(1./cbc.GL[0], 1./cbc.GL[1], 1./cbc.GL[2]);
	logPrintf("\nRlandauer [10^-16 Ohm-m^2]: ");
	(Rlandauer*(1./rhoLambdaUnit)).print(globalLog, " %lg ");
	logPrintf("\n");
	
	//Save ballistic conductance contributions:
	if(mpiWorld->isHead())
	{	writeProfile(cbc.Gzz, rhoLambdaUnit, NkFine, "ballisticConductance.dat");
		writeProfile(cbc.GLz, rhoLambdaUnit, NkFine, "ballisticConductanceL.dat");

		//Compute and output average velocity:
		for(size_t i=0; i<cbc.vNum.size(); i++)
			cbc.vNum[i] /= std::max(cbc.vDen[i], 1E-16); //regularize to v->0 for kx, ky with no Fermi states
		const double nm = 10*Angstrom;
		writeProfile(cbc.vNum, 1./(nm/fs), NkFine, "ballisticConductanceV.dat");
	}
	
	//Cleanup:
	fw.free();
	FeynWann::finalize();
	return 0;
}
