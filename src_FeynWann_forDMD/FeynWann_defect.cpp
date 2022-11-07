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


void FeynWann::defectLoop(const vector3<>& k01, const vector3<>& k02, FeynWann::defectProcessFunc defectProcess, void* params,
	eProcessFunc eProcess1, eProcessFunc eProcess2,
	const std::vector<bool>* eMask1, const std::vector<bool>* eMask2, const std::vector<bool>* defectMask)
{	static StopWatch watchBcast("FeynWann::defectLoop:bcast"); 
	static StopWatch watchCallback("FeynWann::defectLoop:callback");
	assert(fwp.needDefect.length());
	int prodOffsetDim = Hw->nkTot;
	int prodOffsetDimSq = HdefectW->nkTot;
	assert(prodOffsetDimSq == prodOffsetDim*prodOffsetDim);
	
	//Initialize electronic states for 1 and 2:
	#define PrepareElecStates(i) \
		bool withinRange##i = false; \
		std::vector<StateE> e##i(prodOffsetDim); /* States */ \
		{	eTransformNeeded(k0##i); \
			int ik = Hw->ikStart; \
			int ikStop = ik + Hw->nk; \
			PartialLoop3D(offsetDim, ik, ikStop, e##i[ik].k, k0##i, \
				e##i[ik].ik = ik; \
				e##i[ik].withinRange = eMask##i ? eMask##i->at(ik) : true; \
				setState(e##i[ik]); \
				if(e##i[ik].withinRange) \
				{	withinRange##i = true; \
					if(eProcess##i) eProcess##i(e##i[ik], params); \
				} \
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
	PrepareElecStates(1) //prepares e1 and V1
	if(not withinRange1 and not (eProcess2)) return; //no states in active window of 1 and no other callbacks requested
	PrepareElecStates(2) //prepares e2 and V2
	if(not (withinRange1 and withinRange2)) return; //no states in either active window
	#undef PrepareElecStates
	
	//Initialize net mask combining range entries and specified mask (if any):
	std::vector<bool> pairMask(defectMask ? *defectMask : std::vector<bool>(prodOffsetDimSq, true));
	if(fwp.ePhHeadOnly) { pairMask.assign(prodOffsetDimSq, false); pairMask[0] = true; } //only first entry
	auto pairIter = pairMask.begin();
	int nNZ = 0;
	for(int ik1=0; ik1<prodOffsetDim; ik1++)
		for(int ik2=0; ik2<prodOffsetDim; ik2++)
		{	bool netMask = (*pairIter) and e1[ik1].withinRange and e2[ik2].withinRange;
			if(netMask) nNZ++;
			*(pairIter++) = netMask;
		}
	bool bypassTransform = (nNZ <= tTransformByComputeD);
	
	//Calculate electron-defect matrix elements:
	if(bypassTransform)
	{	//Loop over computes, stores data in same locations as transform:
		int ikPair = 0;
		int iProc = 0; //which process should contain this data:
		auto pairIter = pairMask.begin();
		for(int ik1=0; ik1<prodOffsetDim; ik1++)
			for(int ik2=0; ik2<prodOffsetDim; ik2++)
			{	if(*(pairIter++)) HdefectW->compute(e1[ik1].k, e2[ik2].k, ikPair, iProc);
				ikPair++;
				while(iProc+1<mpiGroup->nProcesses() and ikPair==HdefectW->ikStartProc[iProc+1]) iProc++;
			}
	}
	else HdefectW->transform(k01, k02); //generate all data in a single transform
	
	//Process call back function using these matrix elements:
	int ikPair = 0;
	int ikPairStart = HdefectW->ikStart;
	int ikPairStop = ikPairStart + HdefectW->nk;
	int ik1 = 0; vector3<> k1;
	PartialLoop3D(offsetDim, ik1, prodOffsetDim, k1, k01,
		if(e1[ik1].withinRange)
		{	int ik2 = 0; vector3<> k2;
			PartialLoop3D(offsetDim, ik2, prodOffsetDim, k2, k02,
				if(ikPair>=ikPairStart and ikPair<ikPairStop //subset to be evaluated on this process
					and pairMask[ikPair] ) //state pair is active (includes e2.withinRange due to net mask constructed above)
				{	//Set defect matrix elements:
					MatrixDefect m;
					setMatrix(e1[ik1], e2[ik2], ikPair, m);
					//Invoke call-back function:
					watchCallback.start();
					defectProcess(m, params);
					watchCallback.stop();
				}
				ikPair++;
			)
		}
		else ikPair += prodOffsetDim; //no states within range at current k1
	)
}


void FeynWann::defectCalc(const FeynWann::StateE& e1, const FeynWann::StateE& e2, FeynWann::MatrixDefect& m)
{	assert(fwp.needDefect.length());
	//Compute Fourier version of Hdefect for specified k1,k2 pair:
	HdefectW->compute(e1.k, e2.k);
	//Prepare state on group head:
	if(mpiGroup->isHead()) setMatrix(e1, e2, 0, m);
}


void FeynWann::setMatrix(const FeynWann::StateE& e1, const FeynWann::StateE& e2, int ikPair, FeynWann::MatrixDefect& m)
{	static StopWatch watch("FeynWann::setMatrix"); watch.start();
	m.e1 = &e1;
	m.e2 = &e2;
	//Get the (short-ranged) matrix elements:
	m.M = getMatrix(HdefectW->getResult(ikPair), nBands, nBands);
	//TODO: add long range polar corrections if required:
	//Switch to E1 and E2 eigenbasis:
	m.M = dagger(m.e1->U) * m.M * m.e2->U; //to E1 and E2 eigenbasis
	watch.stop();
}

