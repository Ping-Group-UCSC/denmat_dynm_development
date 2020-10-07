/*-------------------------------------------------------------------
Copyright 2019 Ravishankar Sundararaman, Adela Habib

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
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/Units.h>
#include "FeynWann.h"
#include "Histogram.h"
#include "InputMap.h"
#include "LindbladFile.h"

//Reverse iterator for pointers:
template<class T> constexpr std::reverse_iterator<T*> reverse(T* i) { return std::reverse_iterator<T*>(i); }

static const double omegaPhCut = 1e-6;
static const double nEphDelta = 4.; //number of ePhDelta to include in output

//Lindblad initialization using FeynWann callback
struct LindbladInit
{	
	FeynWann& fw;
	const vector3<int>& NkFine; //!< effective k-point mesh sampled
	const size_t nkTot; //!< total k-points effectively used in BZ sampling
	
	const double dmuMin, dmuMax, Tmax;
	const double pumpOmegaMax, pumpTau, probeOmegaMax;
	
	const bool ePhEnabled, ePhOnlyElec, ePhOnlyHole; //!< whether e-ph coupling is enabled
	const double ePhDelta; //!< Gaussian energy conservation width
	
	size_t oStart, oStop; //!< range of offstes handled by this process group
	size_t noMine, oInterval; //!< number of offsets on this process group and reporting interval
	
	
	LindbladInit(FeynWann& fw, const vector3<int>& NkFine,
		double dmuMin, double dmuMax, double Tmax, 
		double pumpOmegaMax, double pumpTau, double probeOmegaMax,
		bool ePhEnabled, bool ePhOnlyElec, bool ePhOnlyHole, double ePhDelta)
		: fw(fw), NkFine(NkFine), nkTot(NkFine[0] * NkFine[1] * NkFine[2]),
		dmuMin(dmuMin), dmuMax(dmuMax), Tmax(Tmax),
		pumpOmegaMax(pumpOmegaMax), pumpTau(pumpTau), probeOmegaMax(probeOmegaMax),
		ePhEnabled(ePhEnabled), ePhOnlyElec(ePhOnlyElec), ePhOnlyHole(ePhOnlyHole), ePhDelta(ePhDelta)
	{
	}
	
	//--------- k-point selection -------------
	
	double EvMax, EcMin, Emid; //VBM and CBM estimates
	inline void eRange(const FeynWann::StateE& state)
	{	for(const double& E: state.E)
		{	if(E<1e-4 and E>EvMax) EvMax = E;
			if(E>1e-4 and E<EcMin) EcMin = E;
		}
	}
	static void eRange(const FeynWann::StateE& state, void* params)
	{	((LindbladInit*)params)->eRange(state);
	}
	
	double Estart, Estop; //energy range for k selection
	std::vector<vector3<>> k; //selected k-points
	std::vector<double> E; //all band energies for selected k-points
	inline void kSelect(const FeynWann::StateE& state){
		bool active = false;
		bool has_elec = false, has_hole = false;
		for(double E: state.E)
			if(E>=Estart and E<=Estop){	
				active = true;
				break;
			}
		for (double E : state.E)
			if (E >= Emid and E <= Estop){
				has_elec = true;
				break;
			}
		for (double E : state.E)
			if (E >= Estart and E <= Emid){
				has_hole = true;
				break;
			}
		if(active){
			if ((!ePhOnlyElec && !ePhOnlyHole) || (ePhOnlyElec && has_elec) || (ePhOnlyHole && has_hole)){
				k.push_back(state.k);
				E.insert(E.end(), state.E.begin(), state.E.end());
			}
		}
	}
	static void kSelect(const FeynWann::StateE& state, void* params)
	{	((LindbladInit*)params)->kSelect(state);
	}
	void kpointSelect(const std::vector<vector3<>>& k0)
	{
		//Determine energy range:
		EvMax = -DBL_MAX;
		EcMin = +DBL_MAX;
		fw.eLoop(vector3<>(), LindbladInit::eRange, this);
		mpiWorld->allReduce(EvMax, MPIUtil::ReduceMax);
		mpiWorld->allReduce(EcMin, MPIUtil::ReduceMin);
		logPrintf("VBM: %.3lf eV, CBM: %.3lf eV\n", EvMax / eV, EcMin / eV);
		Emid = (EvMax + EcMin) / 2.;
		//--- add margins of max phonon energy, energy conservation width and fermiPrime width
		double Emargin =7.*Tmax; //neglect below 10^-3 occupation deviation from equilibrium
		Estart = std::min(EcMin - pumpOmegaMax - 4./pumpTau, std::min(EvMax, dmuMin) - Emargin);
		Estop = std::max(EvMax + pumpOmegaMax + 4./pumpTau, std::max(EcMin, dmuMax) + Emargin);
		logPrintf("Active energy range for kpointSelect: %.3lf to %.3lf eV\n", Estart/eV, Estop/eV);
		
		//Initialize sampling parameters:
		size_t oStart, oStop; //range of offstes handled by this process group
		if (mpiGroup->isHead())
			TaskDivision(k0.size(), mpiGroupHead).myRange(oStart, oStop);
		mpiGroup->bcast(oStart);
		mpiGroup->bcast(oStop);
		size_t noMine = oStop - oStart;
		size_t oInterval = std::max(1, int(round(noMine / 50.))); //interval for reporting progress

		//Select k-points:
		logPrintf("Scanning k-points with active states: "); logFlush();
		for(size_t o=oStart; o<oStop; o++)
		{	fw.eLoop(k0[o], LindbladInit::kSelect, this);
			//Print progress:
			if((o-oStart+1)%oInterval==0) { logPrintf("%d%% ", int(round((o-oStart+1)*100./noMine))); logFlush(); }
		}
		logPrintf("done.\n"); logFlush();
		
		//Synchronize selected k and E across all processes:
		//--- determine nk on each process and compute cumulative counts
		std::vector<size_t> nkPrev(mpiWorld->nProcesses()+1);
		for(int jProc=0; jProc<mpiWorld->nProcesses(); jProc++)
		{	size_t nkCur = k.size();
			mpiWorld->bcast(nkCur, jProc); //nkCur = k.size() on jProc in all processes
			nkPrev[jProc+1] = nkPrev[jProc] + nkCur; //cumulative count
		}
		size_t nkSelected = nkPrev.back();
		//--- broadcast k and E:
		{	//Set k and E in position in global arrays:
			std::vector<vector3<>> k(nkSelected);
			std::vector<double> E(nkSelected*fw.nBands);
			std::copy(this->k.begin(), this->k.end(), k.begin()+nkPrev[mpiWorld->iProcess()]);
			std::copy(this->E.begin(), this->E.end(), E.begin()+nkPrev[mpiWorld->iProcess()]*fw.nBands);
			//Broadcast:
			for(int jProc=0; jProc<mpiWorld->nProcesses(); jProc++)
			{	size_t ikStart = nkPrev[jProc], nk = nkPrev[jProc+1]-ikStart;
				mpiWorld->bcast(k.data()+ikStart, nk, jProc);
				mpiWorld->bcast(E.data()+ikStart*fw.nBands, nk*fw.nBands, jProc);
			}
			//Store to class variables:
			std::swap(k, this->k);
			std::swap(E, this->E);
		}
		logPrintf("Found %lu k-points with active states from %lu total k-points (%.0fx reduction)\n\n",
			nkSelected, nkTot, round(nkTot*1./nkSelected));
	}
	
	//--------- k-pair selection -------------
	std::vector<std::vector<size_t>> kpartners; //list of e-ph coupled k2 for each k1
	std::vector<std::pair<size_t,size_t>> kpairs; //pairs of k1 and k2
	std::map<size_t,size_t> kIndexMap; //map from k-point mesh index to index in selected set
	inline size_t kIndex(vector3<> k)
	{	size_t index=0;
		for(int iDir=0; iDir<3; iDir++)
		{	double ki = k[iDir] - floor(k[iDir]); //wrapped to [0,1)
			index = (size_t)round(NkFine[iDir]*(index+ki));
		}
		return index;
	}
	inline void selectActive(const double*& Ebegin, const double*& Eend, double Elo, double Ehi) //narrow pointer range to data within [Estart,Estop]
	{	Ebegin = std::lower_bound(Ebegin, Eend, Elo);
		Eend = &(*std::lower_bound(reverse(Eend), reverse(Ebegin), Ehi, std::greater<double>()))+1;
	}
	inline void kpSelect(const FeynWann::StatePh& state)
	{	//Find pairs of momentum conserving electron states with this q:
		for (size_t ik1 = 0; ik1<k.size(); ik1++){
			const vector3<>& k1 = k[ik1];
			vector3<> k2 = k1 - state.q; //momentum conservation
			const std::map<size_t, size_t>::iterator iter = kIndexMap.find(kIndex(k2));
			if (iter != kIndexMap.end())
			{
				size_t ik2 = iter->second;
				//Check energy conservation for pair of bands within active range:
				//--- determine ranges of all E1 and E2:
				const double *E1begin = E.data() + ik1*fw.nBands, *E1end = E1begin + fw.nBands;
				const double *E2begin = E.data() + ik2*fw.nBands, *E2end = E2begin + fw.nBands;
				//--- narrow to active energy ranges:
				selectActive(E1begin, E1end, Estart, Estop);
				selectActive(E2begin, E2end, Estart, Estop);
				//--- check energy ranges:
				bool Econserve = false;
				for (const double* E1 = E1begin; E1<E1end; E1++) //E1 in active range
				{
					for (const double* E2 = E2begin; E2<E2end; E2++) //E2 in active range
					{
						for (const double omegaPh : state.omega) if (omegaPh>omegaPhCut) //loop over non-zero phonon frequencies
						{
							double deltaE = (*E1) - (*E2) - omegaPh; //energy conservation violation
							if (fabs(deltaE) < nEphDelta*ePhDelta) //else negligible at the 10^-3 level for a Gaussian
							{
								Econserve = true;
								break;
							}
						}
						if (Econserve) break;
					}
					if (Econserve) break;
				}
				if (Econserve) kpairs.push_back(std::make_pair(ik1, ik2));
			}
		}
	}
	static void kpSelect(const FeynWann::StatePh& state, void* params)
	{
		((LindbladInit*)params)->kpSelect(state);
	}
	void kpairSelect(const std::vector<vector3<>>& q0){
		// reset Estart and Estop. Note that ePhOnlyElec and ePhOnlyHole cannot be both true
		double Estart_save = Estart, Estop_save = Estop;
		if (ePhOnlyElec) { Estart = Emid; }
		if (ePhOnlyHole) { Estop = Emid; }

		//Initialize kIndexMap for searching selected k-points:
		for(size_t ik=0; ik<k.size(); ik++)
			kIndexMap[kIndex(k[ik])] = ik;
		
		//Initialize sampling parameters:
		size_t oStart, oStop; //!< range of offstes handled by this process groups
		if (mpiGroup->isHead())
			TaskDivision(q0.size(), mpiGroupHead).myRange(oStart, oStop);
		mpiGroup->bcast(oStart);
		mpiGroup->bcast(oStop);
		size_t noMine = oStop - oStart;
		size_t oInterval = std::max(1, int(round(noMine / 50.))); //interval for reporting progress

		//Find momentum-conserving k-pairs for which energy conservation is also possible for some bands:
		logPrintf("Scanning k-pairs with e-ph coupling: "); logFlush();
		for (size_t o = oStart; o<oStop; o++){
			fw.phLoop(q0[o], LindbladInit::kpSelect, this);
			//Print progress:
			if ((o - oStart + 1) % oInterval == 0) { logPrintf("%d%% ", int(round((o - oStart + 1)*100. / noMine))); logFlush(); }
		}
		logPrintf("done.\n"); logFlush();
		// After selecting k pairs, Estart and Estop must be set back
		Estart = Estart_save; Estop = Estop_save;

		//Synchronize selected kpairs across all processes:
		//--- determine nk on each process and compute cumulative counts
		std::vector<size_t> nkpPrev(mpiWorld->nProcesses()+1);
		for(int jProc=0; jProc<mpiWorld->nProcesses(); jProc++)
		{	size_t nkpCur = kpairs.size();
			mpiWorld->bcast(nkpCur, jProc); //nkCur = k.size() on jProc in all processes
			nkpPrev[jProc+1] = nkpPrev[jProc] + nkpCur; //cumulative count
		}
		size_t nkpairs = nkpPrev.back();
		//--- broadcast k and E:
		{	//Set k and E in position in global arrays:
			std::vector<std::pair<size_t,size_t>> kpairs(nkpairs);
			std::copy(this->kpairs.begin(), this->kpairs.end(), kpairs.begin()+nkpPrev[mpiWorld->iProcess()]);
			//Broadcast:
			for(int jProc=0; jProc<mpiWorld->nProcesses(); jProc++)
			{	size_t ikpStart = nkpPrev[jProc], nkp = nkpPrev[jProc+1]-ikpStart;
				mpiWorld->bcast(((size_t*)kpairs.data())+ikpStart*2, nkp*2, jProc);
			}
			//Store to class variables:
			std::swap(kpairs, this->kpairs);
		}
		//--- report:
		size_t nkpairsTot = k.size()*k.size();
		logPrintf("Found %lu k-pairs with e-ph coupling from %lu total pairs of selected k-points (%.0fx reduction)\n",
			nkpairs, nkpairsTot, round(nkpairsTot*1./nkpairs));
		//--- initialize kpartners (list of k2 by k1):
		kpartners.resize(k.size());
		for(auto kpair: kpairs)
			kpartners[kpair.first].push_back(kpair.second);
		size_t nPartnersMin = k.size(), nPartnersMax = 0;
		for(std::vector<size_t>& kp: kpartners)
		{	std::sort(kp.begin(), kp.end()); //sort k2 within each k1 array
			const size_t& nPartners = kp.size();
			if(nPartners < nPartnersMin) nPartnersMin = nPartners;
			if(nPartners > nPartnersMax) nPartnersMax = nPartners;
		}
		logPrintf("Number of partners per k-point:  min: %lu  max: %lu  mean: %.1lf\n\n", nPartnersMin, nPartnersMax, nkpairs*1./k.size());
	}

	//--------- Save data -------------
	void saveData()
	{
		//Initialize and write header:
		#ifdef MPI_SAFE_WRITE
		FILE* fp = NULL;
		if(mpiWorld->isHead()) fp = fopen("ldbd.dat", "w"); //I/O from world head alone
		#else
		MPIUtil::File fp;
		if(mpiGroup->isHead()) mpiGroupHead->fopenWrite(fp, "ldbd.dat"); //I/O collectively only from group heads
		#endif
		LindbladFile::Header h;
		h.dmuMin = dmuMin;
		h.dmuMax = dmuMax;
		h.Tmax = Tmax;
		h.pumpOmegaMax = pumpOmegaMax;
		h.probeOmegaMax = probeOmegaMax;
		h.nk = k.size();
		h.nkTot = nkTot;
		h.ePhEnabled = ePhEnabled;
		h.spinorial = (fw.nSpinor==2);
		h.spinWeight = fw.spinWeight;
		h.R = fw.R;
		if(mpiWorld->isHead())
		{	std::ostringstream oss;
			h.write(oss);
			#ifdef MPI_SAFE_WRITE
			fwrite(oss.str().data(), 1, h.nBytes(), fp);
			#else
			mpiGroupHead->fwrite(oss.str().data(), 1, h.nBytes(), fp);
			#endif
		}
		size_t nBytesWritten = h.nBytes();
		
		//Chunk for storing byte offsets to each k-point data before location for data:
		std::vector<size_t> byteOffsets;
		size_t byteOffsetLocation = nBytesWritten; //this is where byteOffsets will be put
		nBytesWritten += sizeof(size_t)*h.nk; //skip ahead now; write at the end once known
		byteOffsets.push_back(nBytesWritten); //offset to first k-point
		
		//Make group index and count available on all processes of each group:
		size_t iGroup, nGroups;
		if(mpiGroup->isHead())
		{	iGroup = mpiGroupHead->iProcess();
			nGroups = mpiGroupHead->nProcesses();
		}
		mpiGroup->bcast(iGroup);
		mpiGroup->bcast(nGroups);
		
		//Loop over k-points in parallel over process groups:
		size_t nPasses = ceildiv(k.size(), nGroups);
		size_t passInterval = std::max(1, int(round(nPasses/50.))); //interval for reporting progress
		logPrintf("Writing ldbd.dat: "); logFlush();
		for(size_t iPass=0; iPass<nPasses; iPass++)
		{	size_t ik = iPass*nGroups + iGroup;
			LindbladFile::Kpoint kp;
			if(ik < k.size())
			{	kp.k = k[ik];
				
				//Determine energy ranges:
				const double *Ebegin = E.data()+ik*fw.nBands, *Eend = Ebegin+fw.nBands;
				//--- pump-active (inner) energy range:
				const double *EinnerBegin = Ebegin, *EinnerEnd = Eend;
				selectActive(EinnerBegin, EinnerEnd, Estart, Estop);
				kp.nInner = EinnerEnd - EinnerBegin;
				//--- probe-active (outer) energy range:
				const double *EouterBegin = Ebegin, *EouterEnd = Eend;
				selectActive(EouterBegin, EouterEnd,
					(*EinnerBegin)-probeOmegaMax, //lowest occupied energy accessible from bottom of active window
					(*(EinnerEnd-1))+probeOmegaMax);  //highest unoccupied energy accessible from top of active window
				kp.nOuter = EouterEnd - EouterBegin;
				kp.innerStart = EinnerBegin - EouterBegin;
				int innerOffset = EinnerBegin - Ebegin; //offset from original bands to inner window
				int outerOffset = EouterBegin - Ebegin; //offset from original bands to outer window
				
				//Calculate electronic state:
				FeynWann::StateE ei;
				fw.eCalc(kp.k, ei);
				
				//Save energy and matrix elements to kp:
				if(mpiGroup->isHead())
				{	//Energies:
					kp.E.assign(EouterBegin, EouterEnd);
					//Momenta:
					for(int iDir=0; iDir<3; iDir++)
						kp.P[iDir] = ei.v[iDir](innerOffset,innerOffset+kp.nInner, outerOffset,outerOffset+kp.nOuter);
					//Spin:
					if(h.spinorial)
						for(int iDir=0; iDir<3; iDir++)
							kp.S[iDir] = ei.S[iDir](innerOffset,innerOffset+kp.nInner, innerOffset,innerOffset+kp.nInner);
				}
				
				//Electron-phonon matrix elements:
				if(h.ePhEnabled)
				{
					for(size_t jk: kpartners[ik])
					{	
						//Compute other electronic state:
						FeynWann::StateE ej;
						fw.eCalc(k[jk], ej);
						//--- determine active range:
						const double *EjBegin = E.data()+jk*fw.nBands, *EjEnd = EjBegin+fw.nBands;
						const double *EjInnerBegin = EjBegin, *EjInnerEnd = EjEnd;
						selectActive(EjInnerBegin, EjInnerEnd, Estart, Estop);
						int innerOffset_j = EjInnerBegin - EjBegin;
						int nInner_j = EjInnerEnd - EjInnerBegin;
						
						//Compute phonon state:
						FeynWann::StatePh ph;
						fw.phCalc(k[ik]-k[jk], ph);
						
						//Compute e-ph matrix elements:
						FeynWann::MatrixEph m;
						fw.ePhCalc(ei, ej, ph, m);
						
						//Collect energy-conserving matrix elements within active window:
						if(mpiGroup->isHead())
						{	for(int alpha=0; alpha<fw.nModes; alpha++)
							{	LindbladFile::GePhEntry g;
								g.jk = jk;
								g.omegaPh = m.ph->omega[alpha];
								if(g.omegaPh < omegaPhCut) continue; //avoid zero frequency phonons
								double sigmaInv = 1./std::min(ePhDelta, g.omegaPh/(nEphDelta+1)); //make sure flipped energies not included within energy conservation
								double deltaPrefac = sqrt(sigmaInv/sqrt(M_PI));
								const matrix& M = m.M[alpha];
								for(int n2=innerOffset_j; n2<innerOffset_j+nInner_j; n2++)
									for(int n1=innerOffset; n1<innerOffset+kp.nInner; n1++)
									{	double deltaEbySigma = sigmaInv*(m.e1->E[n1] - m.e2->E[n2] - g.omegaPh);
										if(fabs(deltaEbySigma) < nEphDelta)
										{	SparseEntry s;
											s.i = n1 - innerOffset;
											s.j = n2 - innerOffset_j;
											s.val = M(n1,n2) * (deltaPrefac*exp(-0.5*deltaEbySigma*deltaEbySigma)); //apply e-conservation factor
											g.G.push_back(s);
										}
									}
								if(g.G.size()) kp.GePh.push_back(g);
							}
						}
					}
				}
			}
			
			//Synchronize write from group heads (or send data to world head if MPI_SAFE_WRITE) to make sure kpoints are written in order:
			if(mpiGroup->isHead())
			{
				for(size_t jGroup=0; jGroup<nGroups; jGroup++)
				{	size_t byteOffsetNext = byteOffsets.back();
					if(jGroup==iGroup and ik<k.size())
						byteOffsetNext += kp.nBytes(h);
					mpiGroupHead->bcast(byteOffsetNext, jGroup);
					byteOffsets.push_back(byteOffsetNext); //available on all group heads for all k
				}
				
				if(ik<k.size())
				{	std::ostringstream oss;
					kp.write(oss, h);
					#ifdef MPI_SAFE_WRITE
					//Send data to world head to write (may be necessary on NFS locations due to MPI-IO issues):
					if(mpiGroupHead->isHead()) //on head already; write:
					{	fseek(fp, byteOffsets[ik], SEEK_SET);
						fwrite(oss.str().data(), 1, kp.nBytes(h), fp);
					}
					else
						mpiGroupHead->send(oss.str().data(), kp.nBytes(h), 0, iGroup);//send to head to write
					#else
					//Write from each process in parallel based on offset determined above:
					mpiGroupHead->fseek(fp, byteOffsets[ik], SEEK_SET);
					mpiGroupHead->fwrite(oss.str().data(), 1, kp.nBytes(h), fp);
					#endif
				}
				#ifdef MPI_SAFE_WRITE
				//Write data from other processes on head:
				if(mpiGroupHead->isHead())
				{	for(size_t jGroup=1; jGroup<nGroups; jGroup++)
					{	size_t ikRemote = iPass*nGroups + jGroup;
						if(ikRemote < k.size())
						{	std::vector<char> buf(byteOffsets[ikRemote+1]-byteOffsets[ikRemote]);
							mpiGroupHead->recvData(buf, jGroup, jGroup); //recv data to write
							fseek(fp, byteOffsets[ikRemote], SEEK_SET);
							fwrite(buf.data(), 1, buf.size(), fp);
						}
					}
				}
				#endif
				
				nBytesWritten = byteOffsets.back();
			}
			
			//Print progress:
			if((iPass+1)%passInterval==0) { logPrintf("%d%% ", int(round((iPass+1)*100./nPasses))); logFlush(); }
		}
		logPrintf("done.\n"); logFlush();
		
		//Write byte offsets:
		if(mpiWorld->isHead())
		{	byteOffsets.resize(h.nk); //drop the end values (only keep start for the actual k's written)
			#ifdef MPI_SAFE_WRITE
			fseek(fp, byteOffsetLocation, SEEK_SET);
			fwrite(byteOffsets.data(), sizeof(size_t), byteOffsets.size(), fp);
			#else
			mpiGroupHead->fseek(fp, byteOffsetLocation, SEEK_SET);
			mpiGroupHead->fwriteData(byteOffsets, fp);
			#endif
		}
		#ifdef MPI_SAFE_WRITE
		if(mpiWorld->isHead()) fclose(fp);
		#else
		if(mpiGroup->isHead()) mpiGroupHead->fclose(fp);
		#endif
	}
};

int main(int argc, char** argv)
{	
	InitParams ip = FeynWann::initialize(argc, argv, "Initialize sparse matrices for Lindblad dynamics");

	//Get the system parameters:
	InputMap inputMap(ip.inputFilename);
	//--- kpoints
	const int NkMultAll = int(round(inputMap.get("NkMult"))); //increase in number of k-points for phonon mesh
	vector3<int> NkMult;
	NkMult[0] = inputMap.get("NkxMult", NkMultAll); //override increase in x direction
	NkMult[1] = inputMap.get("NkyMult", NkMultAll); //override increase in y direction
	NkMult[2] = inputMap.get("NkzMult", NkMultAll); //override increase in z direction
	//--- doping / temperature
	const double dmuMin = inputMap.get("dmuMin", 0.) * eV; //optional: lowest shift in fermi level from neutral value / VBM in eV (default: 0)
	const double dmuMax = inputMap.get("dmuMax", 0.) * eV; //optional: highest shift in fermi level from neutral value / VBM in eV (default: 0)
	const double Tmax = inputMap.get("Tmax") * Kelvin; //maximum temperature in Kelvin (ambient phonon T = initial electron T)
	//--- pump
	const double pumpOmegaMax = inputMap.get("pumpOmegaMax") * eV; //maximum pump frequency in eV
	const double pumpTau = inputMap.get("pumpTau") * fs; //maximum pump frequency in eV
	const double probeOmegaMax = inputMap.get("probeOmegaMax") * eV; //maximum probe frequency in eV
	const string ePhMode = inputMap.getString("ePhMode"); //must be Off or DiagK (add FullK in future)
	const bool ePhEnabled = (ePhMode != "Off");
	const bool ePhOnlyElec = inputMap.get("ePhOnlyElec", 0);
	const bool ePhOnlyHole = inputMap.get("ePhOnlyHole", 0);
	bool notBothTrue = !(ePhOnlyElec && ePhOnlyHole); assert(notBothTrue);
	const double ePhDelta = inputMap.get("ePhDelta") * eV; //energy conservation width for e-ph coupling
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("NkMult = "); NkMult.print(globalLog, " %d ");
	logPrintf("dmuMin = %lg\n", dmuMin);
	logPrintf("dmuMax = %lg\n", dmuMax);
	logPrintf("Tmax = %lg\n", Tmax);
	logPrintf("pumpOmegaMax = %lg\n", pumpOmegaMax);
	logPrintf("pumpTau = %lg\n", pumpTau);
	logPrintf("probeOmegaMax = %lg\n", probeOmegaMax);
	logPrintf("ePhMode = %s\n", ePhMode.c_str());
	logPrintf("ePhOnlyElec = %d\n", ePhOnlyElec);
	logPrintf("ePhOnlyHole = %d\n", ePhOnlyHole);
	logPrintf("ePhDelta = %lg\n", ePhDelta);
	
	//Initialize FeynWann:
	FeynWannParams fwp(&inputMap);
	fwp.needVelocity = true;
	fwp.needSpin = true;
	fwp.needPhonons = ePhEnabled;
	FeynWann fw(fwp);

	//Construct mesh of k-offsets:
	std::vector<vector3<>> k0;
	vector3<int> NkFine;
	for(int iDir=0; iDir<3; iDir++)
	{	if(fw.isTruncated[iDir] && NkMult[iDir]!=1)
		{	logPrintf("Setting NkMult = 1 along truncated direction %d.\n", iDir+1);
			NkMult[iDir] = 1; //no multiplication in truncated directions
		}
		NkFine[iDir] = fw.kfold[iDir] * NkMult[iDir];
	}
	matrix3<> NkFineInv = inv(Diag(vector3<>(NkFine)));
	vector3<int> ikMult;
	for(ikMult[0]=0; ikMult[0]<NkMult[0]; ikMult[0]++)
	for(ikMult[1]=0; ikMult[1]<NkMult[1]; ikMult[1]++)
	for(ikMult[2]=0; ikMult[2]<NkMult[2]; ikMult[2]++)
		k0.push_back(NkFineInv * ikMult);
	logPrintf("Effective interpolated k-mesh dimensions: ");
	NkFine.print(globalLog, " %d ");
	size_t nOffsets = k0.size();
	size_t nKeff = nOffsets * fw.eCountPerOffset();
	logPrintf("Effectively sampled %s: %lu\n", "nKpts", nKeff);

	//Construct mesh of q-offsets:
	std::vector<vector3<>> q0;
	if (ePhEnabled)
	{
		vector3<int> NqMult;
		for (int iDir = 0; iDir<3; iDir++)
			NqMult[iDir] = NkFine[iDir] / fw.phononSup[iDir];
		vector3<int> iqMult;
		for (iqMult[0] = 0; iqMult[0]<NqMult[0]; iqMult[0]++)
		for (iqMult[1] = 0; iqMult[1]<NqMult[1]; iqMult[1]++)
		for (iqMult[2] = 0; iqMult[2]<NqMult[2]; iqMult[2]++)
			q0.push_back(NkFineInv * iqMult);
	}
	
	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		fw.free();
		FeynWann::finalize();
		return 0;
	}
	logPrintf("\n");
	
	//Create and initialize lindblad calculator:
	LindbladInit lb(fw, NkFine, dmuMin, dmuMax, Tmax, pumpOmegaMax, pumpTau, probeOmegaMax, ePhEnabled, ePhOnlyElec, ePhOnlyHole, ePhDelta);

	//First pass (e only): select k-points
	lb.kpointSelect(k0);

	//Second pass (ph only): select k pairs
	if (ePhEnabled)
		lb.kpairSelect(q0);

	//Final pass: output electronic and e-ph quantities
	lb.saveData();
	
	//Cleanup:
	fw.free();
	FeynWann::finalize();
	return 0;
}
