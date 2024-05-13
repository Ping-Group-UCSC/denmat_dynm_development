/*-------------------------------------------------------------------
Copyright 2022 Ravishankar Sundararaman, Josh Quinton

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
#include <wannier/WannierMinimizer.h>

#ifdef TDEP_ENABLED
#include "tdep_wrapper.h"
#endif


void FeynWann::phLoop(const vector3<>& q0, FeynWann::phProcessFunc phProcess, void* params)
{	static StopWatch watchCallback("FeynWann::phLoop:callback");
	assert(fwp.needPhonons);
	//Run Fourier transforms with this offset:
	if(not fwp.tdep) OsqW->transform(q0);
	//Call phProcess for q-points on present process:
	int iq = OsqW->ikStart;
	int iqStop = iq + OsqW->nk;
	StatePh state;
	PartialLoop3D(offsetDim, iq, iqStop, state.q, q0,
		state.iq = iq;
		setState(state);
		watchCallback.start();
		phProcess(state, params);
		watchCallback.stop();
	)
}


void FeynWann::phCalc(const vector3<>& q, FeynWann::StatePh& ph)
{	assert(fwp.needPhonons);
	//Compute Fourier versions for this q:
	if(not fwp.tdep) OsqW->compute(q);
	//Prepare state on group head:
	ph.iq = 0;
	ph.q = q;
	if(mpiGroup->isHead()) setState(ph);
}


void FeynWann::ePhLoop(const vector3<>& k01, const vector3<>& k02, FeynWann::ePhProcessFunc ePhProcess, void* params,
	eProcessFunc eProcess1, eProcessFunc eProcess2, phProcessFunc phProcess,
	const std::vector<bool>* eMask1, const std::vector<bool>* eMask2, const std::vector<bool>* ePhMask)
{	static StopWatch watchBcast("FeynWann::ePhLoop:bcast"); 
	static StopWatch watchCallback("FeynWann::ePhLoop:callback");
	assert(fwp.needPhonons);
	int prodOffsetDim = Hw->nkTot;
	int prodOffsetDimSq = HePhW->nkTot;
	assert(prodOffsetDim == OsqW->nkTot);
	assert(prodOffsetDimSq == prodOffsetDim*prodOffsetDim);
	
	//Initialize electronic states for 1 and 2:
	#define PrepareElecStates(i) \
		bool withinRange##i = false; \
		std::vector<StateE> e##i(prodOffsetDim); /* States */ \
		{	eTransformNeeded(k0##i); \
			HePhSumW->transform(k0##i); \
			Dw->transform(k0##i); \
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
	inEphLoop = true; //turns on sum rule handling in setState and bcastState
	PrepareElecStates(1) //prepares e1 and V1
	if(not withinRange1 and not (eProcess2 or phProcess)) return; //no states in active window of 1 and no other callbacks requested
	PrepareElecStates(2) //prepares e2 and V2
	if(not (withinRange1 and withinRange2) and not phProcess) return; //no states in either active window (and no other callbacks requested)
	inEphLoop = false;
	#undef PrepareElecStates
	
	//Prepare phonon states:
	vector3<> q0 = k01 - k02;
	if(not fwp.tdep) OsqW->transform(q0);
	std::vector<StatePh> ph(prodOffsetDim);
	{	int iq = OsqW->ikStart;
		int iqStop = iq + OsqW->nk;
		PartialLoop3D(offsetDim, iq, iqStop, ph[iq].q, q0,
			ph[iq].iq = iq;
			setState(ph[iq]);
			if(phProcess) phProcess(ph[iq], params);
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
	if(not (withinRange1 and withinRange2)) return; //no pairs of states within active window
	
	//Initialize net mask combining range entries and specified mask (if any):
	std::vector<bool> pairMask(ePhMask ? *ePhMask : std::vector<bool>(prodOffsetDimSq, true));
	if(fwp.ePhHeadOnly) { pairMask.assign(prodOffsetDimSq, false); pairMask[0] = true; } //only first entry
	auto pairIter = pairMask.begin();
	int nNZ = 0;
	for(int ik1=0; ik1<prodOffsetDim; ik1++)
		for(int ik2=0; ik2<prodOffsetDim; ik2++)
		{	bool netMask = (*pairIter) and e1[ik1].withinRange and e2[ik2].withinRange;
			if(netMask) nNZ++;
			*(pairIter++) = netMask;
		}
	bool bypassTransform = (nNZ <= tTransformByCompute);
	
	//Calculate electron-phonon matrix elements:
	if(bypassTransform)
	{	//Loop over computes, stores data in same locations as transform:
		int ikPair = 0;
		int iProc = 0; //which process should contain this data:
		auto pairIter = pairMask.begin();
		for(int ik1=0; ik1<prodOffsetDim; ik1++)
			for(int ik2=0; ik2<prodOffsetDim; ik2++)
			{	if(*(pairIter++)) HePhW->compute(e1[ik1].k, e2[ik2].k, ikPair, iProc);
				ikPair++;
				while(iProc+1<mpiGroup->nProcesses() and ikPair==HePhW->ikStartProc[iProc+1]) iProc++;
			}
	}
	else HePhW->transform(k01, k02); //generate all data in a single transform
	
	//Process call back function using these matrix elements:
	int ikPair = 0;
	int ikPairStart = HePhW->ikStart;
	int ikPairStop = ikPairStart + HePhW->nk;
	int ik1 = 0; vector3<> k1;
	PartialLoop3D(offsetDim, ik1, prodOffsetDim, k1, k01,
		if(e1[ik1].withinRange)
		{	int ik2 = 0; vector3<> k2;
			PartialLoop3D(offsetDim, ik2, prodOffsetDim, k2, k02,
				if(ikPair>=ikPairStart and ikPair<ikPairStop //subset to be evaluated on this process
					and (not (fwp.ePhHeadOnly and ikPair)) //overridden in k-path debug mode to be ikPair==0 alone
					and pairMask[ikPair] ) //state pair is active (includes e2.withinRange due to net mask constructed above)
				{	//Identify associated phonon states:
					int iqIndex = calculateIndex(ik1v - ik2v, offsetDim);
					//Set e-ph matrix elements:
					MatrixEph m;
					setMatrix(e1[ik1], e2[ik2], ph[iqIndex], ikPair, m);
					//Invoke call-back function:
					watchCallback.start();
					ePhProcess(m, params);
					watchCallback.stop();
				}
				ikPair++;
			)
		}
		else ikPair += prodOffsetDim; //no states within range at current k1
	)
}


void FeynWann::ePhCalc(const FeynWann::StateE& e1, const FeynWann::StateE& e2, const FeynWann::StatePh& ph, FeynWann::MatrixEph& m)
{	assert(fwp.needPhonons);
	assert(circDistanceSquared(e1.k-e2.k, ph.q) < 1e-8);
	//Compute Fourier version of HePh for specified k1,k2 pair:
	HePhW->compute(e1.k, e2.k);
	//Prepare state on group head:
	if(mpiGroup->isHead()) setMatrix(e1, e2, ph, 0, m);
}


void FeynWann::setState(FeynWann::StatePh& state)
{	assert(fwp.needPhonons);
	
	if(fwp.tdep)
	{
		#ifdef TDEP_ENABLED
		vector3<> qCart = GT * -state.q / (2*M_PI); //TDEP uses Cartesian q
		state.omega.resize(nModes);
		state.U = zeroes(nModes, nModes);
		tdep_compute_(&qCart[0], state.omega.data(), state.U.data());
		return;
		#endif
	}
	
	//Get force matrix:
	matrix Osqq = getMatrix(OsqW->getResult(state.iq), nModes, nModes);

	//Add polar corrections (LO-TO  splits) if any:
	if(polar and state.q.isNonzero())
	{	//Prefactor including denominator:
		int prodSup = OsqW->nkTot;
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
	if(polar and ph.q.isNonzero())
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
	//const double omegaPhCut = 1e-6;
	m.M.resize(nModes);
	for(int iMode=0; iMode<nModes; iMode++)
		m.M[iMode] = sqrt(m.ph->omega[iMode]<omegaPhCut ? 0. : 0.5/m.ph->omega[iMode]) //frequency-dependent phonon amplitude
			* (dagger(m.e1->U) * getMatrix(Mall.data(), nBands, nBands, iMode) * m.e2->U); //to E1 and E2 eigenbasis
	watch.stop();
}
