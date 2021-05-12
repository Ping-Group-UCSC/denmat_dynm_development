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
	const double T;
	const double prefacImSigma;
	const double EconserveExpFac, EconservePrefac; //energy conserving Gaussian exponential and pre-factor
	const bool valley; //whether to add valley contributions
	const unsigned nP; //number of ImSigma's calculated (linewidth, momentum-relaxation, and if available, valley-relaxation)
	const std::vector<double>& f1grid; //grid of fillings ofr which e-ph linewidth is calculated
	std::vector<std::vector<diagMatrix>> ImSigma; //e-ph linewidth for various f1, without and with momentum/valley factors
	std::vector<diagMatrix> E; //save electron energies on DFT mesh for final outputs
	std::vector<vector3<>> kmesh; //DFT k-point mesh (full version i.e. unreduced)
	double wOffsetCur; //weight factor of current offset (due to symmetry reduction)
	std::vector<vector3<int>> uniqueCells; //for wannierization of results
	const matrix3<> G, GGT; 
	const vector3<> K, Kp; //For computing valley weights
	
	CollectEph(const FeynWann& fw, double T, double EconserveWidth, const vector3<int>& NkMult, bool valley)
	: fw(fw), T(T),
		prefacImSigma(0.5 * 2*M_PI/(prod(fw.kfold)*prod(NkMult))), //Factor of 0.5 in ImSigma because of psi^2 -> n
		EconserveExpFac(-0.5/std::pow(EconserveWidth,2)),
		EconservePrefac(1./(sqrt(2.*M_PI)*EconserveWidth)),
		valley(valley),
		nP(valley ? 3 : 2),
		f1grid(FeynWannParams::fGrid_ePh),
		ImSigma(nP*f1grid.size(), std::vector<diagMatrix>(prod(fw.kfold), diagMatrix(fw.nBands))),
		E(prod(fw.kfold)), kmesh(prod(fw.kfold)),
		G(2*M_PI * inv(fw.R)), GGT(G * (~G)),
		K(1./3, 1./3, 0), Kp(-1./3, -1./3, 0)
	{
		//Check lattice compatibility with valley mode
		if(valley)
		{	double cosTheta12 = dot(fw.R.column(0),fw.R.column(1))/(fw.R.column(0).length()*fw.R.column(1).length());
			double cosTheta13 = dot(fw.R.column(0),fw.R.column(2))/(fw.R.column(0).length()*fw.R.column(2).length());
			double cosTheta23 = dot(fw.R.column(1),fw.R.column(2))/(fw.R.column(1).length()*fw.R.column(2).length());
			if(fabs(cosTheta12+0.5)>symmThreshold or fabs(cosTheta13)>symmThreshold or fabs(cosTheta23)>symmThreshold)
			{	((FeynWann&)fw).free();
				FeynWann::finalize();
				die("Valley mode requires hexagonal lattice with 120 degrees between first two lattice vectors.\n\n");
			}
		}
	}
	
	//Helper functions for evaluating inter-valley weights (Hex crystals only):
	static inline vector3<> wrap(const vector3<>& x)
	{	vector3<> result = x;
		for(int dir=0; dir<3; dir++)
			result[dir] -= floor(0.5 + result[dir]);
		return result;
	}
	inline bool isKvalley(vector3<> k) const
	{	return GGT.metric_length_squared(wrap(K-k))
			< GGT.metric_length_squared(wrap(Kp-k));
	} 
	
	//---- Collect energies and kmesh ----
	static void collectE(const FeynWann::StateE& state, void* params)
	{	CollectEph& cEph = *((CollectEph*)params);
		int ik = calculateIndex(round(Diag(state.k)*cEph.fw.kfold), cEph.fw.kfold);
		cEph.E[ik] = state.E;
		cEph.kmesh[ik] = state.k;
	}
	
	//---- Main e-ph scattering linewidth kernel ----
	void process(const FeynWann::MatrixEph& mEph)
	{	const FeynWann::StateE& e1 = *(mEph.e1);
		const FeynWann::StateE& e2 = *(mEph.e2);
		const FeynWann::StatePh& ph = *(mEph.ph);
		const int ik1 = calculateIndex(round(Diag(e1.k)*fw.kfold), fw.kfold);
		const int nBands = e1.E.nRows();
		const int nModes = ph.omega.nRows();
		
		//Weight for valley contrib
		double wValley = (valley and (isKvalley(e1.k) xor isKvalley(e2.k))) ? 1. : 0.;
		
		//Loop over electronic state 1:
		for(int b1=0; b1<nBands; b1++)
		{	const double& E1 = e1.E[b1];
			const vector3<>& v1 = e1.vVec[b1];
			//Loop over electronic state 2:
			for(int b2=0; b2<nBands; b2++)
			{	const double& E2 = e2.E[b2];
				const vector3<>& v2 = e2.vVec[b2];
				double cosThetaScatter = dot(v1, v2) / sqrt(std::max(1e-16, v1.length_squared() * v2.length_squared()));
				//Loop over phonon modes:
				for(int alpha=0; alpha<nModes; alpha++)
				{	const double& omegaPh = ph.omega[alpha];
					double omegaPhByT = omegaPh/T;
					if(omegaPhByT < 1e-3) continue; //avoid 0./0. below
					double nPh = bose(omegaPhByT);
					if(!nPh) continue; //no contribution below
					//Loop over absorption and emission:
					for(int ae=-1; ae<=+1; ae+=2)
					{	double EconserveExponent = EconserveExpFac * std::pow((E2-E1 - ae*omegaPh),2);
						if(EconserveExponent < -15.) continue; //the exponential below will be negligible
						double delta = EconservePrefac * exp(EconserveExponent);
						double contribNum = wOffsetCur * prefacImSigma * delta * mEph.M[alpha](b2,b1).norm()
							* nPh*(nPh+1); //contribution numerator before f1-dependent denominator
						for(unsigned if1=0; if1<f1grid.size(); if1++)
						{	unsigned if1p = if1 + f1grid.size(); //index for scattering version
							unsigned if1v = if1p + f1grid.size(); //index for valley version (if present)
							double contrib = contribNum / (nPh+0.5 + ae*(0.5-f1grid[if1])); //net f1-dependent contribution
							ImSigma[if1][ik1][b1] += contrib;
							ImSigma[if1p][ik1][b1] += contrib * (1.-cosThetaScatter); //scattering version with angle factors
							if(valley) ImSigma[if1v][ik1][b1] += contrib * wValley; //scattering intervalley contribution
						}
					}
				}
			}
		}
	}
	static void ePhProcess(const FeynWann::MatrixEph& mEph, void* params)
	{	((CollectEph*)params)->process(mEph);
	}
	
	//---- Wannierization ----
	int cStart, cStop; //range of cells handled here
	int iCol; //current column
	matrix mlwfImSigma[3], phase;
	void wannierize(const FeynWann::StateE& state)
	{	const int ik = calculateIndex(round(Diag(state.k)*fw.kfold), fw.kfold);
		//Calculate phase for Fourier transform:
		for(int c=cStart; c<cStop; c++)
			phase.set(iCol, c-cStart, cis(-2*M_PI*dot(state.k, uniqueCells[c])));
		//For each matrix:
		for(unsigned iP=0; iP<nP; iP++) //without or with P factors
		{	for(unsigned if1=0; if1<f1grid.size(); if1++)
			{	//Convert to log for the interpolation:
				diagMatrix logImSigma(ImSigma[if1+iP*f1grid.size()][ik]);
				for(double& x: logImSigma) x = log(x);
				//Switch to Wannier basis:
				matrix logImSigmaW = state.U * logImSigma * dagger(state.U);
				//Save as a column in a matrix containing all k:
				unsigned colLength = fw.nBands * fw.nBands;
				eblas_copy(mlwfImSigma[iP].data()+colLength*(iCol*f1grid.size()+if1), logImSigmaW.data(), colLength);
			}
		}
		iCol++;
	}
	static void eProcess(const FeynWann::StateE& state, void* params)
	{	((CollectEph*)params)->wannierize(state);
	}
	
	void dumpWannierized(matrix& m, string fname) const
	{	m = m * phase; //Fourier transform
		mpiGroup->allReduce(m.data(), m.nData(), MPIUtil::ReduceSum); //Collect results within groups
		if(mpiGroup->isHead())
		{	//expand to all cells version (with zeroes where unavailable currently)
			matrix mEx = zeroes(m.nRows(), uniqueCells.size());
			if(cStop>cStart)
				mEx.set(0,m.nRows(), cStart,cStop, m);
			//Collect results between group heads
			mpiGroupHead->allReduce(mEx.data(), mEx.nData(), MPIUtil::ReduceSum);
			//Output from world head:
			if(mpiGroupHead->isHead())
				mEx.dump(fname.c_str(), fw.realPartOnly); //Output
		}
	}
};


//Report ImSigma for N states closest to Fermi level
class FermiImSigmaReport
{	const size_t N;
	std::multimap<double, std::pair<double,double>> cache;
public:
	FermiImSigmaReport(size_t N) : N(N) {}
	
	void addState(double E, double ImSigma)
	{	auto entry = std::make_pair(fabs(E), std::make_pair(E, ImSigma));
		if(cache.size() < N)
			cache.insert(entry);
		else
		{	if(entry.first < cache.rbegin()->first)
			{	//current one better than worst entry in cache
				cache.erase(--cache.end()); //remove worst entry
				cache.insert(entry); //add currnet one
			}
		}
	}
	
	void report() const
	{	for(auto entry: cache)
			logPrintf("\t%+9.6lf %14.12lf\n", entry.second.first, entry.second.second);
	}
};

int main(int argc, char** argv)
{   InitParams ip =  FeynWann::initialize(argc, argv, "Electron-phonon scattering contribution to electron linewidth.");

	//Read input file:
	InputMap inputMap(ip.inputFilename);
	const double T = inputMap.get("T") * Kelvin;
	const double EconserveWidth = inputMap.get("EconserveWidth") * eV;
	const int iSpin = inputMap.get("iSpin", 0); //spin channel (default 0)
	const int NkMultAll = int(round(inputMap.get("NkMult"))); //increase in number of k-points for phonon mesh
	const string valleyMode = inputMap.has("valley") ? inputMap.getString("valley") : "no"; //whether to also compute intervalley-weighted linewidths
	if(valleyMode!="yes" and valleyMode!="no") die("\nvalleyMode must be 'yes' or 'no'\n");
	const bool valley = (valleyMode!="no");
	FeynWannParams fwp(&inputMap);
	
	vector3<int> NkMult;
	NkMult[0] = inputMap.get("NkxMult", NkMultAll); //override increase in x direction
	NkMult[1] = inputMap.get("NkyMult", NkMultAll); //override increase in y direction
	NkMult[2] = inputMap.get("NkzMult", NkMultAll); //override increase in z direction
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("T = %lg\n", T);
	logPrintf("EconserveWidth = %lg\n", EconserveWidth);
	logPrintf("iSpin = %d\n", iSpin);
	logPrintf("NkMult = "); NkMult.print(globalLog, " %d ");
	logPrintf("valley = %s\n", valleyMode.c_str());
	fwp.printParams();
	
	//Initialize FeynWann:
	fwp.iSpin = iSpin;
	fwp.needSymmetries = true;
	fwp.needPhonons = true;
	fwp.needVelocity = true;
	fwp.needSpin = true;
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
	vector3<> kOffset;
	vector3<int> NkFine;
	for(int iDir=0; iDir<3; iDir++)
	{	kOffset[iDir] = fw.isTruncated[iDir] ? 0. : 0.5; //offset from Gamma in periodic directions
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
		kMult.push_back(NkMultInv * (ikMult + kOffset));
	logPrintf("Effective interpolated k-mesh dimensions: ");
	NkFine.print(globalLog, " %d ");
	
	//Collect energies and k-point  mesh:
	CollectEph cEph(fw, T, EconserveWidth, NkMult, valley);
	for(vector3<> qOff: fw.qOffset) fw.eLoop(qOff, CollectEph::collectE, &cEph);
	//--- make available on all processes:
	for(unsigned i=0; i<cEph.E.size(); i++)
	{	int root = cEph.E[i].size() ? mpiGroup->iProcess() : mpiGroup->nProcesses(); //my process ID or N, depending on whether I have E[i]
		mpiGroup->allReduce(root, MPIUtil::ReduceMin); //lowest process number which has E[i] available
		cEph.E[i].resize(fw.nBands);
		mpiGroup->bcast(cEph.E[i].data(), fw.nBands, root);
	}
	mpiGroup->allReduce(&cEph.kmesh[0][0], 3*cEph.kmesh.size(), MPIUtil::ReduceSum);
	
	//Estimate minimum NkMult:
	{	//--- Get density of states:
		TetrahedralDOS dosEval(cEph.kmesh, std::vector<int>(), fw.R, Diag(fw.kfold), 1, fw.nBands, 1);
		double Etol = 1e-4;
		dosEval.setEigs(cEph.E);
		dosEval.weldEigenvalues(Etol);
		TetrahedralDOS::Lspline dos = dosEval.gaussSmooth(dosEval.getDOS(0, Etol), EconserveWidth); //dos within EconserveWidth of each energy
		//--- Find minimum DOS at each DFT eigenvalue
		double Ecut = 5*eV; //only include states ~ 5 eV from fermi level / VBM
		double dosMin = DBL_MAX;
		double E0 = dos[0].first;
		double dEinv = 1./(dos[1].first-E0);
		int ikStart, ikStop; TaskDivision(cEph.E.size(), mpiWorld).myRange(ikStart, ikStop);
		for(int ik=ikStart; ik<ikStop; ik++)
			for(double E: cEph.E[ik])
				if(fabs(E) < Ecut)
				{	double t = dEinv*(E-E0);
					int i = floor(t); t -= i;
					double dosCur = dos[i].second[0]*(1.-t) + dos[i+1].second[0]*t;
					dosMin = std::min(dosMin, dosCur);
				}
		mpiWorld->allReduce(dosMin, MPIUtil::ReduceMin);
		//--- Calculate minimum folding such that minimum average states per EconserveWidth
		double prodNkFineMin = 100./(dosMin * EconserveWidth);
		int nDim = 0;
		for(int iDir=0; iDir<3; iDir++)
			if(NkFine[iDir]>1)
				nDim++; //only count directions which have more than one k-point
		int NkMultDim = round(std::pow(prodNkFineMin/cEph.kmesh.size(), 1./nDim));
		vector3<int> NkMultMin;
		for(int iDir=0; iDir<3; iDir++)
			NkMultMin[iDir] = (NkFine[iDir]>1) ? NkMultDim : 1;
		logPrintf("\nFor 100 states within EconserveWidth, NkMult ~ ");
		NkMultMin.print(globalLog, " %d ");
	}
	
	//Reduce under symmetries (simplified version of Symmetries::reduceKmesh from JDFTx):
	std::vector<vector3<>> k02; //array of k2-mesh offsets
	std::vector<double> wk02; //corresponding weights
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
				k02.push_back(kfoldInv * kMult[iSrc]);
				wk02.push_back(w);
			}
	}
	//--- make available on all processes
	int nOffsets = k02.size(); mpiWorld->bcast(nOffsets);
	k02.resize(nOffsets); mpiWorld->bcastData(k02);
	wk02.resize(nOffsets); mpiWorld->bcastData(wk02);
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
	logPrintf("Collecting ImSigma_ePh: "); logFlush();
	for(int oPair=oPairStart; oPair<oPairStop; oPair++)
	{	int o = oPair / nqOffsetSq;
		int iqOff1 = (oPair - o * nqOffsetSq) / nqOffset;
		int iqOff2 = oPair % nqOffset;
		//Process with selected offset:
		cEph.wOffsetCur = wk02[o];
		fw.ePhLoop(fw.qOffset[iqOff1], fw.qOffset[iqOff2] + k02[o], CollectEph::ePhProcess, &cEph);
		//Print progress:
		if((oPair-oPairStart+1)%oPairInterval==0) { logPrintf("%d%% ", int(round((oPair-oPairStart+1)*100./noPairsMine))); logFlush(); }
	}
	logPrintf("done.\n"); logFlush();
	
	//Collect results from all processes:
	for(std::vector<diagMatrix>& dArr: cEph.ImSigma)
		for(diagMatrix& d: dArr)
			mpiWorld->allReduceData(d, MPIUtil::ReduceSum);
	
	//Symmetrize:
	PeriodicLookup<vector3<>> plook(cEph.kmesh, GGT);
	std::vector<bool> kDone(cEph.kmesh.size(), false);
	std::vector<int> iReduced;
	for(size_t i0=0; i0<cEph.kmesh.size(); i0++)
		if(!kDone[i0])
		{	//Find orbit of this k-points under symmetries:
			std::vector<int> iEquiv;
			std::vector<diagMatrix> ImSigmaMean(cEph.ImSigma.size(), diagMatrix(fw.nBands));
			for(int invert: invertList)
				for(const SpaceGroupOp& op: fw.sym)
				{	size_t i = plook.find(invert * cEph.kmesh[i0] * op.rot);
					if(i!=string::npos && (!kDone[i]))
					{	kDone[i] = true; //i will be covered in i0's orbit
						iEquiv.push_back(i);
						for(unsigned iMat=0; iMat<ImSigmaMean.size(); iMat++)
							ImSigmaMean[iMat] += cEph.ImSigma[iMat][i];
					}
				}
			//Symmetrize within orbit:
			for(unsigned iMat=0; iMat<ImSigmaMean.size(); iMat++)
			{	ImSigmaMean[iMat] *= (1./iEquiv.size());
				for(int i: iEquiv)
					cEph.ImSigma[iMat][i] = ImSigmaMean[iMat];
			}
			iReduced.push_back(i0);
		}
	logPrintf("Symmetrized ImSigma for %lu k-points in mesh in %lu orbits.\n", cEph.kmesh.size(), iReduced.size());
	
	//Output linewidths and energies in text file:
	if(mpiWorld->isHead())
	{	FermiImSigmaReport fr(10);
		string fname = "ImSigma_ePh" + fw.spinSuffix + ".dat";
		logPrintf("Dumping '%s' ... ", fname.c_str()); fflush(globalLog);
		FILE* fp = fopen(fname.c_str(), "w");
		for(int i: iReduced)
			for(int b=0; b<fw.nBands; b++)
			{	fprintf(fp, "%+16.12lf", cEph.E[i][b]);
				for(unsigned iMat=0; iMat<cEph.ImSigma.size(); iMat++)
					fprintf(fp, " %19.12le", cEph.ImSigma[iMat][i][b]);
				fprintf(fp, "\n");
				fr.addState(cEph.E[i][b], cEph.ImSigma[0][i][b]);
			}
		fclose(fp);
		logPrintf("done.\n");
		logPrintf("\nEnergy and ImSigma [Eh] for few states closest to Fermi level:\n");
		fr.report();
		logPrintf("HINT: check convergence of above numbers with NkMult.\n\n");
	}
	
	//Wannierize output:
	//--- create unique cells:
	cEph.uniqueCells.reserve(cEph.kmesh.size());
	{	vector3<int> iR;
		for(iR[0]=0; iR[0]<fw.kfold[0]; iR[0]++)
		for(iR[1]=0; iR[1]<fw.kfold[1]; iR[1]++)
		for(iR[2]=0; iR[2]<fw.kfold[2]; iR[2]++)
			cEph.uniqueCells.push_back(iR);
		assert(cEph.uniqueCells.size() == cEph.kmesh.size());
	}
	//--- divide output cells over MPI groups:
	cEph.cStart = cEph.cStop = 0;
	if(mpiGroup->isHead())
		TaskDivision(cEph.kmesh.size(), mpiGroupHead).myRange(cEph.cStart, cEph.cStop);
	mpiGroup->bcast(cEph.cStart);
	mpiGroup->bcast(cEph.cStop);
	int ncMine = std::max(1, cEph.cStop - cEph.cStart);
	int nkMine = std::max(1, fw.Hw->nk * nqOffset);
	//--- Wannierize
	for(unsigned iP=0; iP<cEph.nP; iP++)
		cEph.mlwfImSigma[iP] = zeroes(fw.nBands*fw.nBands*cEph.f1grid.size(), nkMine);
	cEph.phase = zeroes(nkMine, ncMine);
	cEph.iCol = 0;
	for(vector3<> qOff: fw.qOffset) fw.eLoop(qOff, CollectEph::eProcess, &cEph);
	cEph.phase *= (1./cEph.kmesh.size()); //inverse transform normalizing factor
	cEph.dumpWannierized(cEph.mlwfImSigma[0], fwp.wannierPrefix + ".mlwfImSigma_ePh" + fw.spinSuffix);
	cEph.dumpWannierized(cEph.mlwfImSigma[1], fwp.wannierPrefix + ".mlwfImSigmaP_ePh" + fw.spinSuffix);
	if(valley) cEph.dumpWannierized(cEph.mlwfImSigma[2], fwp.wannierPrefix + ".mlwfImSigmaV_ePh" + fw.spinSuffix);
	
	fw.free();
	FeynWann::finalize();
	return 0;
}
