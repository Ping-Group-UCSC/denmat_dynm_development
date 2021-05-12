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
#include <core/Units.h>
#include <core/LatticeUtils.h>
#include <electronic/TetrahedralDOS.h>
#include <algorithm>

template<typename T> T prod(const vector3<T>& v) { return v[0]*v[1]*v[2]; }

struct CollectEph
{	
	const FeynWann& fw;
	const double dmu;
	const vector3<> q0;
	const double prefacG, prefacDOS;
	const double EconserveExpFac, EconservePrefac; //energy conserving (fermi-surface constraining) Gaussian exponential and pre-factor
	std::vector<diagMatrix> G; //Fermi-surface integrated e-ph coupling (for each phonon mode on DFT electronic k-mesh)
	std::vector<diagMatrix> Gp; //Same as G, except weighted by velocity change factor (momentum-relaxation version)
	double g; //Density of states / unit cell
	matrix3<> vv; //Outer product of velocities
	std::vector<diagMatrix> omegaPh; //save phonon energies on full qmesh for final outputs
	std::vector<vector3<>> qmesh; //phonon q-mesh (full version i.e. unreduced)
	double wOffsetCur; //weight factor of current offset (due to symmetry reduction)
	
	CollectEph(const FeynWann& fw, double EconserveWidth, const vector3<int>& NkMult, double dmu, vector3<> q0)
	: fw(fw), dmu(dmu), q0(q0),
		prefacG(fw.spinWeight * 2*M_PI/(prod(fw.kfold)*prod(NkMult))),
		prefacDOS(fw.spinWeight * 1./(2*prod(fw.kfoldSup)*prod(fw.kfold)*prod(NkMult))), //2 prod(kfoldSup) to account for multiple counting
		EconserveExpFac(-0.5/std::pow(EconserveWidth,2)),
		EconservePrefac(1./(sqrt(2.*M_PI)*EconserveWidth)),
		G(prod(fw.kfold), diagMatrix(fw.nModes)),
		Gp(prod(fw.kfold), diagMatrix(fw.nModes)),
		g(0.),
		omegaPh(prod(fw.kfold), diagMatrix(fw.nModes)),
		qmesh(prod(fw.kfold))
	{
	}
	
	//Calculate Fermi-surface delta function and set hasContrib=true if any non-zero
	diagMatrix delta(diagMatrix E)
	{	diagMatrix result(E.nRows());
		for(int b=0; b<E.nRows(); b++)
		{	double deltaExponent = EconserveExpFac * std::pow(E[b]-dmu, 2);
			if(deltaExponent < -15.) continue; //the exponential below will be negligible
			result[b] = EconservePrefac * exp(deltaExponent);
		}
		return result;
	}
	
	//---- Main Fermi-surface-integrated e-ph coupling kernel ----
	void process(const FeynWann::MatrixEph& mEph)
	{	const FeynWann::StateE& e1 = *(mEph.e1);
		const FeynWann::StateE& e2 = *(mEph.e2);
		const FeynWann::StatePh& ph = *(mEph.ph);
		//Svae phonon wave-vectors and frequencies for final outputs
		int iqFine = calculateIndex(round(Diag(ph.q-q0) * fw.kfold), fw.kfold); //referenced to electronic mesh
		qmesh[iqFine] = ph.q;
		omegaPh[iqFine] = ph.omega;
		//Calculate Fermi-surface-constraining delta functions:
		diagMatrix delta1 = delta(e1.E);
		diagMatrix delta2 = delta(e2.E);
		//Collect DOS and related (function of single k):
		#define collectDOS(i, iOther) \
			if(!e##iOther.ik) /*avoid multiple counting due to other k-point*/ \
			{	for(int b=0; b<fw.nBands; b++) if(delta##i[b]) \
				{	const vector3<>& v = e##i.vVec[b]; \
					double contrib = wOffsetCur * prefacDOS * delta##i[b]; \
					g += contrib; \
					vv += contrib * outer(v, v); \
				} \
			}
		collectDOS(1, 2) //collect on e1 k-mesh (offset)
		collectDOS(2, 1) //collect on e2 k-mesh (Gamma-centered)
		//Collect e-ph coupling weight (function of both k's):
		for(int b1=0; b1<fw.nBands; b1++) if(delta1[b1])
		{	const vector3<>& v1 = e1.vVec[b1];
			for(int b2=0; b2<fw.nBands; b2++) if(delta2[b2])
			{	double contrib = wOffsetCur * prefacG * delta1[b1] * delta2[b2];
				const vector3<>& v2 = e2.vVec[b2];
				double cosThetaScatter = dot(v1, v2) / sqrt(std::max(1e-16, v1.length_squared() * v2.length_squared()));
				//Loop over phonon modes:
				for(int alpha=0; alpha<fw.nModes; alpha++)
				{	G[iqFine][alpha] += contrib * mEph.M[alpha](b2,b1).norm();
					Gp[iqFine][alpha] += contrib * mEph.M[alpha](b2,b1).norm() * (1.-cosThetaScatter);
				}
			}
		}
	}
	static void ePhProcess(const FeynWann::MatrixEph& mEph, void* params)
	{	((CollectEph*)params)->process(mEph);
	}
};

int main(int argc, char** argv)
{   InitParams ip =  FeynWann::initialize(argc, argv, "Electron-phonon scattering contribution to phonon linewidth.");

	//Read input file:
	InputMap inputMap(ip.inputFilename);
	const double EconserveWidth = inputMap.get("EconserveWidth") * eV;
	const double dmu = inputMap.get("dmu", 0.) * eV; //shift in electron Fermi level in eV (default: 0 => neutral)
	const int iSpin = inputMap.get("iSpin", 0); //spin channel (default 0)
	const int NkMultAll = int(round(inputMap.get("NkMult"))); //increase in number of k-points for phonon mesh
	vector3<int> NkMult;
	NkMult[0] = inputMap.get("NkxMult", NkMultAll); //override increase in x direction
	NkMult[1] = inputMap.get("NkyMult", NkMultAll); //override increase in y direction
	NkMult[2] = inputMap.get("NkzMult", NkMultAll); //override increase in z direction
	FeynWannParams fwp(&inputMap);
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("EconserveWidth = %lg\n", EconserveWidth);
	logPrintf("dmu = %lg\n", dmu);
	logPrintf("iSpin = %d\n", iSpin);
	logPrintf("NkMult = "); NkMult.print(globalLog, " %d ");
	fwp.printParams();
	
	//Initialize FeynWann:
	fwp.iSpin = iSpin;
	fwp.needSymmetries = true;
	fwp.needPhonons = true;
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
	vector3<> q0; //phonon q-mesh offset
	for(int iDir=0; iDir<3; iDir++)
	{	q0[iDir] = fw.isTruncated[iDir] ? 0. : 0.5/fw.kfold[iDir]; //offset from Gamma in periodic directions
		if(fw.isTruncated[iDir] && NkMult[iDir]!=1)
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
	std::vector<vector3<>> k0; //array of electron k-mesh offsets
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
		vector3<> q0mult = Diag(q0) * fw.kfold;
		for(size_t iSrc=0; iSrc<kMult.size(); iSrc++)
			if(!kDone[iSrc])
			{	double w = 0.; //weight of current offset
				for(int invert: invertList)
					for(const SpaceGroupOp& op: fw.sym)
						for(int swap=0; swap<2; swap++) //whether symmetry maps k and offset k meshes separately, or to each other
						{	size_t iDest = plook.find(invert * kMult[iSrc] * op.rot - q0mult*swap);
							size_t iDestOff = plook.find(invert * (kMult[iSrc] + q0mult) * op.rot - q0mult*(1-swap));
							if(iDest!=string::npos && (!kDone[iDest]) && iDestOff==iDest)
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
	int nqOffset = fw.qOffset.size();
	int nqOffsetSq = nqOffset * nqOffset;
	int nOffsetPairs = nOffsets * nqOffsetSq;
	if(mpiWorld->isHead()) logPrintf("%d phonon q-mesh offset pairs parallelized over %d process groups.\n", nOffsetPairs, mpiGroupHead->nProcesses());
	
	logPrintf("\n");
	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		fw.free();
		FeynWann::finalize();
		return 0;
	}
	
	//Initialize sampling parameters:
	int oPairStart=0, oPairStop=0;
	if(mpiGroup->isHead())
		TaskDivision(nOffsetPairs, mpiGroupHead).myRange(oPairStart, oPairStop);
	mpiGroup->bcast(oPairStart);
	mpiGroup->bcast(oPairStop);
	int noPairsMine = oPairStop-oPairStart; //number of offset pairs handled by current group
	int oPairInterval = std::max(1, int(round(noPairsMine/50.))); //interval for reporting progress
	
	//Collect results for each offset
	logPrintf("Collecting Gph: "); logFlush();
	CollectEph cEph(fw, EconserveWidth, NkMult, dmu, q0);
	for(int oPair=oPairStart; oPair<oPairStop; oPair++)
	{	int o = oPair / nqOffsetSq;
		int iqOff1 = (oPair - o * nqOffsetSq) / nqOffset;
		int iqOff2 = oPair % nqOffset;
		//Process with selected offset:
		cEph.wOffsetCur = wk0[o];
		vector3<> k01 = k0[o] + fw.qOffset[iqOff1] + q0;
		vector3<> k02 = k0[o] + fw.qOffset[iqOff2];
		fw.ePhLoop(k01, k02, CollectEph::ePhProcess, &cEph);
		//Print progress:
		if((oPair-oPairStart+1)%oPairInterval==0) { logPrintf("%d%% ", int(round((oPair-oPairStart+1)*100./noPairsMine))); logFlush(); }
	}
	logPrintf("done.\n"); logFlush();
	
	//Collect results from all processes:
	for(diagMatrix& g: cEph.G) mpiWorld->allReduceData(g, MPIUtil::ReduceSum);
	for(diagMatrix& g: cEph.Gp) mpiWorld->allReduceData(g, MPIUtil::ReduceSum);
	mpiWorld->allReduce(cEph.g, MPIUtil::ReduceSum);
	mpiWorld->allReduce(&cEph.vv(0,0), 9, MPIUtil::ReduceSum);
	for(diagMatrix& o: cEph.omegaPh) mpiWorld->allReduceData(o, MPIUtil::ReduceMax);
	mpiWorld->allReduceData(cEph.qmesh, MPIUtil::ReduceMax);
	
	//Symmetrize:
	PeriodicLookup<vector3<>> plook(cEph.qmesh, GGT);
	std::vector<bool> kDone(cEph.qmesh.size(), false);
	std::vector<int> iReduced;
	std::vector<vector3<>> qReduced;
	std::vector<double> qWeight;
	for(size_t i0=0; i0<cEph.qmesh.size(); i0++)
		if(!kDone[i0])
		{	//Find orbit of this k-points under symmetries:
			std::vector<int> iEquiv;
			diagMatrix Gmean(fw.nModes), GpMean(fw.nModes);
			for(int invert: invertList)
				for(const SpaceGroupOp& op: fw.sym)
				{	size_t i = plook.find(invert * cEph.qmesh[i0] * op.rot);
					if(i!=string::npos && (!kDone[i]))
					{	kDone[i] = true; //i will be covered in i0's orbit
						iEquiv.push_back(i);
						Gmean += cEph.G[i];
						GpMean += cEph.Gp[i];
					}
				}
			//Symmetrize within orbit:
			Gmean *= (1./iEquiv.size());
			GpMean *= (1./iEquiv.size());
			for(int i: iEquiv)
			{	cEph.G[i] = Gmean;
				cEph.Gp[i] = GpMean;
			}
			iReduced.push_back(i0);
			qReduced.push_back(cEph.qmesh[i0]);
			qWeight.push_back(iEquiv.size()*(1./cEph.qmesh.size()));
		}
	logPrintf("Symmetrized Gph for %lu k-points in mesh in %lu orbits.\n", cEph.qmesh.size(), iReduced.size());
	fw.symmetrize(cEph.vv); //symmetrize vSq
	
	//Output linewidths and energies in text file:
	if(mpiWorld->isHead())
	{	string fname = "Gph" + fw.spinSuffix + ".dat";
		logPrintf("Dumping '%s' ... ", fname.c_str()); fflush(globalLog);
		FILE* fp = fopen(fname.c_str(), "w");
		fprintf(fp, "#omegaPh[Eh] G Gp\n");
		for(int i: iReduced)
			for(int b=0; b<fw.nModes; b++)
				fprintf(fp, "%+16.12lf %16.12lf %16.12lf\n", cEph.omegaPh[i][b], cEph.G[i][b], cEph.Gp[i][b]);
		fclose(fp);
		logPrintf("done.\n");
		//q-mesh and weights:
		fname = "Gph" + fw.spinSuffix + ".qList";
		logPrintf("Dumping '%s' ... ", fname.c_str()); fflush(globalLog);
		fp = fopen(fname.c_str(), "w");
		fprintf(fp, "#q0 q1 q2 wq\n");
		for(size_t i=0; i<qReduced.size(); i++)
			fprintf(fp, "%12.10lf %12.10lf %12.10lf  %14.12lf\n",
				qReduced[i][0], qReduced[i][1], qReduced[i][2], qWeight[i]);
		fclose(fp);
		logPrintf("done.\n");
	}
	
	//Report overall moments (used for AC conductivity calculations):
	logPrintf("\nFermi level integrals in atomic units:\n");
	logPrintf("gEf = %lf\n", cEph.g);
	logPrintf("vvEf:\n"); cEph.vv.print(globalLog, " %lf ");
	logPrintf("Omega = %lf\n", fabs(det(fw.R))); //unit cell volume
	logPrintf("\n");
	
	fw.free();
	FeynWann::finalize();
	return 0;
}
