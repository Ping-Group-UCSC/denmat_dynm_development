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
#include <core/LatticeUtils.h>
#include <core/Units.h>


void FeynWann::eLoop(const vector3<>& k0, FeynWann::eProcessFunc eProcess, void* params, const std::vector<bool>* mask)
{	static StopWatch watchCallback("FeynWann::eLoop:callback");
	//Run Fourier transforms with this offset:
	eTransformNeeded(k0);
	//Call eProcess for k-points on present process:
	int ik = Hw->ikStart;
	int ikStop = ik + Hw->nk;
	StateE state;
	PartialLoop3D(offsetDim, ik, ikStop, state.k, k0,
		state.ik = ik;
		state.withinRange = mask ? mask->at(ik) : true;
		setState(state);
		if(state.withinRange)
		{	watchCallback.start();
			eProcess(state, params);
			watchCallback.stop();
		}
	)
}


void FeynWann::eCalc(const vector3<>& k, FeynWann::StateE& e)
{	//Compute Fourier versions for this k:
	eComputeNeeded(k);
	if(fwp.needPhonons) //prepare sum rule quantities
	{	inEphLoop = true;
		Dw->compute(k);
		HePhSumW->compute(k);
	}
	//Prepare state on group head:
	e.ik = 0;
	e.k = k;
	e.withinRange = true;
	if(mpiGroup->isHead()) setState(e);
	inEphLoop = false;
}


void FeynWann::eTransformNeeded(const vector3<>& k0){
	Hw->transform(k0);
	if (!energyOnly){
		if (fwp.needVelocity or (fwp.needL or fwp.needQ)) Pw->transform(k0);
		if (fwp.needSpin) Sw->transform(k0);
		if (fwp.needRP()) { RPw->transform(k0); for (int iDir = 0; iDir < 3; iDir++) HprimeW[iDir]->transform(k0); }
		if (fwp.EzExt) Zw->transform(k0);
		if (fwp.needLayer) Layerw->transform(k0);//JX
		if (fwp.needLinewidth_ee) ImSigma_eeW->transform(k0);
		if (fwp.needLinewidth_ePh) ImSigma_ePhW->transform(k0);
		if (fwp.needLinewidthP_ePh) ImSigmaP_ePhW->transform(k0);
		if (fwp.needLinewidth_D.length()) ImSigma_DW->transform(k0);
		if (fwp.needLinewidthP_D.length()) ImSigmaP_DW->transform(k0);
	}
	else{
		if ((fwp.Bext.isNonzero() or fwp.Bin_model != "none") and fwp.needSpin) Sw->transform(k0);
		if (fwp.Bext.isNonzero() and fwp.orbitalZeeman){
			if (fwp.needVelocity or(fwp.needL or fwp.needQ)) Pw->transform(k0);
			if (fwp.needRP()) { RPw->transform(k0); for (int iDir = 0; iDir < 3; iDir++) HprimeW[iDir]->transform(k0); }
		}
		if (fwp.EzExt) Zw->transform(k0);
	}
}


void FeynWann::eComputeNeeded(const vector3<>& k){
	Hw->compute(k);
	if (!energyOnly){
		if (fwp.needVelocity or(fwp.needL or fwp.needQ)) Pw->compute(k);
		if (fwp.needSpin) Sw->compute(k);
		if (fwp.needRP()) { RPw->compute(k); for (int iDir = 0; iDir < 3; iDir++) HprimeW[iDir]->compute(k); }
		if (fwp.EzExt) Zw->compute(k);
		if (fwp.needLayer) Layerw->compute(k);//JX
		if (fwp.needLinewidth_ee) ImSigma_eeW->compute(k);
		if (fwp.needLinewidth_ePh) ImSigma_ePhW->compute(k);
		if (fwp.needLinewidthP_ePh) ImSigmaP_ePhW->compute(k);
		if (fwp.needLinewidth_D.length()) ImSigma_DW->compute(k);
		if (fwp.needLinewidthP_D.length()) ImSigmaP_DW->compute(k);
	}
	else{
		if ((fwp.Bext.isNonzero() or fwp.Bin_model != "none") and fwp.needSpin) Sw->compute(k);
		if (fwp.Bext.isNonzero() and fwp.orbitalZeeman){
			if (fwp.needVelocity or(fwp.needL or fwp.needQ)) Pw->compute(k);
			if (fwp.needRP()) { RPw->compute(k); for (int iDir = 0; iDir < 3; iDir++) HprimeW[iDir]->compute(k); }
		}
		if (fwp.EzExt) Zw->compute(k);
	}
}


inline void enforceKramerDeg(diagMatrix& E)
{	for(int b=0; b<E.nRows(); b+=2)
	{	double Emid = 0.5*(E[b] + E[b+1]);
		E[b] = Emid;
		E[b+1] = Emid;
	}
}


inline void rotateMatrix(matrix& mat, const matrix& U)
{	mat = dagger(U) * mat * U;
}

//JX
bool FeynWann::is_state_withinRange(StateE& e){
	//Check whether any states in range (only if not already masked out by initial value of withinRange):
	if (e.withinRange and inEphLoop and(ePhEstart<ePhEstop)){
		e.withinRange = false;
		for (double& E : e.E)
		if (E >= ePhEstart and E <= ePhEstop){
			e.withinRange = true;
			break;
		}
	}
	return e.withinRange;
}
//JX

void FeynWann::setState(FeynWann::StateE& state)
{
	static StopWatch watchRotations1("FeynWann::setState:rotations1"),
		watchBL("FeynWann::setState:BL"),
		watchRotations2("FeynWann::setState:rotations2");

	//Get and diagonalize Hamiltonian:
	matrix Hk = getMatrix(Hw->getResult(state.ik), nBands, nBands);

	//JX
	if (Bso.length_squared() == 0) Bso = calc_Bso(state.k);
	bool stark = fwp.EzExt;
	bool zeeman_BS = fwp.needSpin and fwp.Bext.length_squared();
	bool zeeman_BsoS = fwp.needSpin and Bso.length_squared();
	bool zeeman_BL = fwp.Bext.length_squared() and fwp.orbitalZeeman;
	bool hasField = stark or zeeman_BS or zeeman_BsoS or zeeman_BL;

	if (fwp.enforceKramerDeg and hasField){
		matrix U; diagMatrix E;
		Hk.diagonalize(U, E);
		enforceKramerDeg(E);
		Hk = U * E * dagger(U);
	}

	//Stark and BS Perturbations:
	if (stark) Hk += fwp.EzExt * getMatrix(Zw->getResult(state.ik), nBands, nBands);

	/*
	if (zeeman_BS and not zeeman_BL)
	{	//Add S part of Zeeman perturbation:
	vector3<> Bg = fwp.Bext * gfac + Bso * gElectron;
	for (int iDir = 0; iDir < 3; iDir++)
	if (Bg[iDir]) Hk += bohrMagneton * Bg[iDir] * 0.5 * getMatrix(Sw->getResult(state.ik), nBands, nBands, iDir);
	}
	*/
	if (zeeman_BS and not zeeman_BL)
	{	//Add S part of Zeeman perturbation:
		vector3<> Bg = fwp.Bext * gfac;
		for (int iDir = 0; iDir < 3; iDir++)
		if (Bg[iDir]) Hk += bohrMagneton * Bg[iDir] * 0.5 * getMatrix(Sw->getResult(state.ik), nBands, nBands, iDir);
	}

	Hk.diagonalize(state.U, state.E);
	if (fwp.enforceKramerDeg and !hasField) enforceKramerDeg(state.E);

	for (double& E : state.E) E -= mu; //reference to Fermi level
	if (fwp.scissor) fwp.applyScissor(state.E); //assume CBM > 0 and VBM <= 0, which may not be true if there is a field
	if (fwp.scissor_above_band)	{ for (int i = fwp.band_scissor; i < nBands; i++) state.E[i] += fwp.scissor_above_band; } //JX

	//Compute matrix elements
	watchRotations1.start();
	if (!energyOnly){
		bool needV = fwp.needVelocity or (fwp.needL or fwp.needQ);
		if (needV) { for (int iDir = 0; iDir < 3; iDir++) state.v[iDir] = complex(0, -1) * state.getMatrixRotated(Pw, iDir); } //Since P was stored with -i omitted (to make it real when possible)
		if (fwp.needSpin) { for (int iDir = 0; iDir < 3; iDir++) state.S[iDir] = state.getMatrixRotated(Sw, iDir); }
		if (fwp.needL or fwp.needQ) state.computeLQ(fwp, RPw, HprimeW);
	}
	else{
		if (zeeman_BS or zeeman_BsoS) { for (int iDir = 0; iDir < 3; iDir++) state.S[iDir] = state.getMatrixRotated(Sw, iDir); }
		if (zeeman_BL) { for (int iDir = 0; iDir < 3; iDir++) state.v[iDir] = complex(0, -1) * state.getMatrixRotated(Pw, iDir); } //Since P was stored with -i omitted (to make it real when possible)
		if (zeeman_BL) state.computeLQ(fwp, RPw, HprimeW);
	}
	watchRotations1.stop();

	//e-ph sum rule if needed:
	if (!energyOnly and inEphLoop) state.compute_dHePhSum(Dw, HePhSumW); //e-ph sum rule on unperturbed H

	//Bso S Perturbation:
	if (zeeman_BsoS and not zeeman_BL){
		matrix Hpert(state.E);

		for (int iDir = 0; iDir < 3; iDir++)
			if (Bso[iDir]) Hpert += (Bso[iDir] * bohrMagneton * gElectron * 0.5) * restrictBandRange(state.S[iDir], fwp.bStart_model, fwp.bEnd_model);  //0.5 because |S| in [0, 1]

		//Diagonalize perturbed Hamiltonian:
		matrix Upert; //additional rotations due to perturbation
		Hpert.diagonalize(Upert, state.E); //energies are now perturbed
		if (!energyOnly) state.U = state.U * Upert; //eigenvectors are now perturbed

		if (!energyOnly){
			//Apply rotations to matrix elements computed above:
			if (fwp.needVelocity) for (int iDir = 0; iDir<3; iDir++) rotateMatrix(state.v[iDir], Upert);
			if (fwp.needSpin) for (int iDir = 0; iDir<3; iDir++) rotateMatrix(state.S[iDir], Upert);
			if (fwp.needL) for (int iDir = 0; iDir<3; iDir++) rotateMatrix(state.L[iDir], Upert);
			if (fwp.needQ) for (int iComp = 0; iComp<5; iComp++) rotateMatrix(state.Q[iComp], Upert);
		}
	}

	//BL or B(L+2S) Perturbations:
	if (zeeman_BL){
		watchBL.start();
		matrix Hpert(state.E);

		for (int iDir = 0; iDir < 3; iDir++){
			if (zeeman_BS and fwp.Bext[iDir]) Hpert += (fwp.Bext[iDir] * bohrMagneton * gElectron * 0.5) * state.S[iDir];  //0.5 because |S| in [0, 1]
			if (zeeman_BsoS and Bso[iDir]) Hpert += (Bso[iDir] * bohrMagneton * gElectron * 0.5) * restrictBandRange(state.S[iDir], fwp.bStart_model, fwp.bEnd_model);  //0.5 because |S| in [0, 1]
			if (fwp.Bext[iDir])	Hpert += (fwp.Bext[iDir] * bohrMagneton) * restrictInnerWindow(state.L[iDir], state.E);
		}

		//Diagonalize perturbed Hamiltonian:
		matrix Upert; //additional rotations due to perturbation
		Hpert.diagonalize(Upert, state.E); //energies are now perturbed
		if (!energyOnly) state.U = state.U * Upert; //eigenvectors are now perturbed

		if (!energyOnly){
			//Apply rotations to matrix elements computed above:
			if (fwp.needVelocity) for (int iDir = 0; iDir<3; iDir++) rotateMatrix(state.v[iDir], Upert);
			if (fwp.needSpin) for (int iDir = 0; iDir<3; iDir++) rotateMatrix(state.S[iDir], Upert);
			if (fwp.needL) for (int iDir = 0; iDir<3; iDir++) rotateMatrix(state.L[iDir], Upert);
			if (fwp.needQ) for (int iComp = 0; iComp<5; iComp++) rotateMatrix(state.Q[iComp], Upert);
		}
		watchBL.stop();
	}

	Bso = vector3<>(0,0,0); // should be reset to zero

	if (energyOnly) return;
	//Check whether any states in range (only if not already masked out by initial value of withinRange):
	if (not is_state_withinRange(state)) return; //Remaining quantities will never be used
	//JX
		
	//Extract diagonal components for convenience, where needed:
	if(fwp.needVelocity) StateE::extractDiagonal(state.v, state.vVec);
	if(fwp.needSpin) StateE::extractDiagonal(state.S, state.Svec);
	
	//Linewidths, as needed:
	watchRotations2.start();
	if(fwp.needLinewidth_ee)
		state.ImSigma_ee = diag(state.getMatrixRotated(ImSigma_eeW));
	if(fwp.needLinewidth_ePh)
	{	state.logImSigma_ePhArr.resize(FeynWannParams::fGrid_ePh.size());
		for(unsigned iMat=0; iMat<state.logImSigma_ePhArr.size(); iMat++)
			state.logImSigma_ePhArr[iMat] = diag(state.getMatrixRotated(ImSigma_ePhW, iMat));
	}
	if(fwp.needLinewidthP_ePh)
	{	state.logImSigmaP_ePhArr.resize(FeynWannParams::fGrid_ePh.size());
		for(unsigned iMat=0; iMat<state.logImSigmaP_ePhArr.size(); iMat++)
			state.logImSigmaP_ePhArr[iMat] = diag(state.getMatrixRotated(ImSigmaP_ePhW, iMat));
	}
	if(fwp.needLinewidth_D.length())
	{	state.ImSigma_D = diag(state.getMatrixRotated(ImSigma_DW));
		for(double& ImSigma: state.ImSigma_D) ImSigma = exp(ImSigma); //ImSigma_D is interpolated logarithmically
	}
	if(fwp.needLinewidthP_D.length())
	{	state.ImSigmaP_D = diag(state.getMatrixRotated(ImSigmaP_DW));
		for(double& ImSigma: state.ImSigmaP_D) ImSigma = exp(ImSigma); //ImSigmaP_D is interpolated logarithmically
	}
	watchRotations2.stop();
}


void FeynWann::bcastState(FeynWann::StateE& state, MPIUtil* mpiUtil, int root)
{	if(mpiUtil->nProcesses()==1) return; //no communictaion needed
	mpiUtil->bcast(state.ik, root);
	mpiUtil->bcast(&state.k[0], 3, root);
	//Energy and eigenvectors:
	bcast(state.E, nBands, mpiUtil, root);
	mpiUtil->bcast(state.withinRange, root);
	if(not state.withinRange) return; //Remaining quantities will never be used
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
	//Angular momentum matrix, if needed:
	if(fwp.needL)
	{	for(int iDir=0; iDir<3; iDir++)
			bcast(state.L[iDir], nBands, nBands, mpiUtil, root);
	}
	//Electric quadrupole r*p matrix, if needed:
	if(fwp.needQ)
	{	for(int iComp=0; iComp<5; iComp++)
			bcast(state.Q[iComp], nBands, nBands, mpiUtil, root);
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
	if(fwp.needLinewidth_D.length()) bcast(state.ImSigma_D, nBands, mpiUtil, root);
	if(fwp.needLinewidthP_D.length()) bcast(state.ImSigmaP_D, nBands, mpiUtil, root);
	//e-ph sum rule if needed
	if(inEphLoop)
		bcast(state.dHePhSum, nBands*nBands, 3, mpiUtil, root);
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


matrix FeynWann::StateE::getMatrixRotated(const std::shared_ptr<DistributedMatrix>& mat, int iMat) const
{	int nBands = U.nRows();
	return dagger(U) * getMatrix(mat->getResult(ik), nBands, nBands, iMat) * U;
}


void FeynWann::StateE::computeLQ(const FeynWannParams& fwp,
	const std::shared_ptr<DistributedMatrix> RPw,
	const std::shared_ptr<DistributedMatrix> HprimeW[3])
{
	int nBands = E.nRows();
	matrix3<matrix> RP;
	for(int iDir=0; iDir<3; iDir++)
	{
		if(fwp.bandSumLQ)
		{	//Compute RP from sum over bands:
			//--- compute r = i p / Delta E:
			matrix ri = complex(0, -1) * v[iDir];
			complex* riData = ri.data();
			for(int bCol=0; bCol<nBands; bCol++) //note: column major storage
			for(int bRow=0; bRow<nBands; bRow++)
			{	double Ediff = E[bRow] - E[bCol];
				*(riData++) *= (fabs(Ediff) < fwp.degeneracyThreshold ? 0. : 1./Ediff);
			}
			//--- set r * p
			for(int jDir=0; jDir<3; jDir++)
				RP(iDir, jDir) = ri * v[jDir];
		}
		else
		{	//Compute RP from Wannier interpolation with range-splitting:
			for(int jDir=0; jDir<3; jDir++)
				RP(iDir, jDir) = complex(0,-1) //Since RP was stored with -i omitted (to make it real when possible)
					* getMatrixRotated(RPw, 3*iDir+jDir);
			//Long range correction:
			//--- fetch dH/dk
			matrix iDi = getMatrixRotated(HprimeW[iDir]);
			//--- convert to i*D := i dU/dk in place:
			{	complex* iDiData = iDi.data();
				for(int bCol=0; bCol<nBands; bCol++) //note: column major storage
					for(int bRow=0; bRow<nBands; bRow++)
					{	double Ediff = E[bCol] - E[bRow];
						*(iDiData++) *= complex(0., (fabs(Ediff) < fwp.degeneracyThreshold) ? 0. : 1./Ediff);
					}
			}
			//--- add correction
			for(int jDir=0; jDir<3; jDir++)
				RP(iDir, jDir) += iDi * v[jDir];
		}
	}
	
	//Extract L if needed:
	if(fwp.needL)
	{	for(int kDir=0; kDir<3; kDir++)
		{	int iDir = (kDir + 1) % 3;
			int jDir = (kDir + 2) % 3;
			L[kDir] = dagger_symmetrize(RP(iDir, jDir) - RP(jDir, iDir));
		}
	}
	
	//Extract Q if needed:
	if(fwp.needQ)
	{	//xy, yz and zx components:
		for(int iDir=0; iDir<3; iDir++)
		{	int jDir = (iDir + 1) % 3;
			Q[iDir] = dagger_symmetrize(RP(iDir, jDir) + RP(jDir, iDir));
		}
		//xx - r^2/3 and yy - r^2/3 components:
		matrix traceTerm = (1./3) * trace(RP);
		for(int iDir=0; iDir<2; iDir++)
			Q[iDir+3] = 2.*dagger_symmetrize(RP(iDir, iDir) - traceTerm);
	}
}


void FeynWann::StateE::compute_dHePhSum(const std::shared_ptr<DistributedMatrix> Dw,
	const std::shared_ptr<DistributedMatrix> HePhSumW)
{
	int nBands = E.nRows();
	dHePhSum.init(nBands*nBands, 3);
	complex* dHsumData = dHePhSum.dataPref();
	for(int iDir=0; iDir<3; iDir++)
	{	matrix D = getMatrixRotated(Dw, iDir);
		matrix H = getMatrixRotated(HePhSumW, iDir);
		//Compute error in the sum rule:
		const double Emag = 1e-3; //damp correction for energy differences >> Emag (to handle fringes of Wannier window)
		const double expFac = -1./(Emag*Emag);
		complex* Hdata = H.data();
		const complex* Ddata = D.data();
		for(int b2=0; b2<nBands; b2++)
			for(int b1=0; b1<nBands; b1++)
			{	double E12 = E[b1] - E[b2];
				*Hdata -= (*(Ddata++)) * E12;
				*Hdata *= exp(expFac*E12*E12); //damp correction based on energy difference
				Hdata++;
			}
		//Rotate back to Wannier basis and store to HePhSum
		H = U * H * dagger(U);
		callPref(eblas_copy)(dHsumData, H.data(), H.nData());
		dHsumData += H.nData();
	}
}


void FeynWann::StateE::extractDiagonal(const matrix (&X)[3], std::vector<vector3<>>& Xvec)
{	int nBands = X[0].nRows();
	Xvec.resize(nBands);
	for(int iDir=0; iDir<3; iDir++)
	{	for(int b=0; b<nBands; b++)
			Xvec[b][iDir] = X[iDir](b,b).real();
	}
}


matrix FeynWann::restrictInnerWindow(const matrix& mat, const diagMatrix& E) const
{	//Construct mask for whether outside window:
	std::vector<bool> isOutside;
	for(double Ei: E)
		isOutside.push_back((Ei < EminInner) or (Ei > EmaxInner));
	//Project:
	matrix M(mat);
	complex* Mdata = M.data();
	for(int bCol=0; bCol<nBands; bCol++) //note: column major storage
	for(int bRow=0; bRow<nBands; bRow++)
	{	if(isOutside[bCol] or isOutside[bRow])
			*Mdata = 0.;
		Mdata++;
	}
	return M;
}
