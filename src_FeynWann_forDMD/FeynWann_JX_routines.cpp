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
#include "lindbladInit_for-DMD-4.5.6/help.h"

matrix FeynWann::restrictBandRange(const matrix& mat, const int& bStart, const int& bEnd) const
{	matrix M(mat);
	if (bStart >= bEnd) return M;
	complex* Mdata = M.data();
	for (int bCol = 0; bCol<nBands; bCol++) //note: column major storage
	for (int bRow = 0; bRow<nBands; bRow++){
		if (bCol < bStart || bCol >= bEnd || bRow < bStart || bRow >= bEnd)
			*Mdata = 0.;
		Mdata++;
	}
	return M;
}

vector3<> FeynWann::calc_Bso(vector3<> k){
	vector3<> r(0, 0, 0);
	if (fwp.Bin_model == "none" || fwp.Bin_model == "read") return r;
	vector3<> k_cart = wrap(k) * ~G;
	double kl = k_cart.length();
	double fac = 1. / (exp((kl - fwp.kcut_model) / fwp.kcut_width_model / fwp.kcut_model) + 1);
	if (fwp.Bin_model == "3d-rashba" || fwp.Bin_model == "3d-rb")
		r = (fac * fwp.alpha_Bin / sqrt(2)) * vector3<>(k_cart[1] - k_cart[2], k_cart[2] - k_cart[0], k_cart[0] - k_cart[1]);
	else if (fwp.Bin_model == "2d-rashba" || fwp.Bin_model == "2d-rb") // along z
		r = (fac * fwp.alpha_Bin) * cross(k_cart, fwp.dir_2drb); //r = (fac * fwp.alpha_Bin) * vector3<>(k_cart[1], -k_cart[0], 0);
	else if (fwp.Bin_model == "psh"){
		r = (fac * fwp.alpha_Bin * dot(k_cart, fwp.pshk)) * fwp.pshs;
	}
	//logPrintf("fac = %lg ; r = %lg %lg %lg ; |r| = %lg\n", fac, r[0], r[1], r[2], r.length());
	return r;
}

//Copy assignment
inline void copy(const matrix& in, matrix& out){
	out.init(in.nRows(), in.nCols(), in.isOnGpu());
	memcpy((ManagedMemory<complex>&)out, (const ManagedMemory<complex>&)in);
}

//Prepare and broadcast matrices on custom communicator:
inline void bcast(diagMatrix& m, int nRows, MPIUtil* mpiUtil, int root){
	m.resize(nRows);
	mpiUtil->bcast(m.data(), nRows, root);
}
inline void bcast(matrix& m, int nRows, int nCols, MPIUtil* mpiUtil, int root){
	if (!m) m.init(nRows, nCols);
	mpiUtil->bcast(m.data(), m.nData(), root);
}
inline void bcast(diagMatrix& m, MPIUtil* mpiUtil, int root){
	int nRows = m.nRows();
	mpiUtil->bcast(nRows, root);
	m.resize(nRows);
	mpiUtil->bcast(m.data(), nRows, root);
}
inline void bcast(matrix& m, MPIUtil* mpiUtil, int root){
	int nRows = m.nRows(), nCols = m.nCols();
	mpiUtil->bcast(nRows, root); mpiUtil->bcast(nCols, root);
	if(!m) m.init(nRows, nCols);
	mpiUtil->bcast(m.data(), m.nData(), root);
}
inline void bcast(std::vector<vector3<>>& v, MPIUtil* mpiUtil, int root){
	int nv = v.size();
	mpiUtil->bcast(nv, root);
	v.resize(nv);
	mpiUtil->bcastData(v, root);
}

void FeynWann::copy_stateE(const FeynWann::StateE& e, FeynWann::StateE& eout){
	eout.ik = e.ik;
	eout.k = e.k;
	eout.withinRange = e.withinRange;
	eout.E = e.E(0, e.E.size());
	if (energyOnly) return;
	copy(e.U, eout.U);
	if (fwp.needVelocity){
		eout.vVec.resize(e.vVec.size());
		for (size_t iDir = 0; iDir < 3; iDir++){
			copy(e.v[iDir], eout.v[iDir]);
			for (size_t b = 0; b < e.vVec.size(); b++)
				eout.vVec[b][iDir] = e.vVec[b][iDir];
		}
	}
	if (fwp.needSpin){
		eout.Svec.resize(0);
		for (size_t iDir = 0; iDir < 3; iDir++)
			copy(e.S[iDir], eout.S[iDir]);
	}
	if (fwp.needL){
		for (size_t iDir = 0; iDir < 3; iDir++)
			copy(e.L[iDir], eout.L[iDir]);
	}
	if (fwp.needQ){
		for (int iComp = 0; iComp < 5; iComp++)
			copy(e.Q[iComp], eout.Q[iComp]);
	}
	if (fwp.needLayer) copy(e.layer, eout.layer);
	//do not need linewidths
	//do not copy dHePhSum //copy(e.dHePhSum, eout.dHePhSum);
}

void FeynWann::trunc_stateE(FeynWann::StateE& e, FeynWann::StateE& eTrunc, int b0_eph, int b1_eph, int b0_dm, int b1_dm, int b0_probe, int b1_probe){
	eTrunc.ik = e.ik;
	eTrunc.k = e.k;
	eTrunc.withinRange = e.withinRange;
	eTrunc.E = e.E(b0_probe, b1_probe);
	if (energyOnly) return;
	copy(e.U(0, nBands, b0_eph, b1_eph), eTrunc.U);
	if (fwp.needVelocity){
		eTrunc.vVec.resize(b1_eph - b0_eph);
		for (size_t iDir = 0; iDir < 3; iDir++){
			copy(e.v[iDir](b0_dm, b1_dm, b0_probe, b1_probe), eTrunc.v[iDir]);
			for (int b = b0_eph; b < b1_eph; b++)
				eTrunc.vVec[b - b0_eph][iDir] = e.v[iDir](b, b).real();
		}
	}
	if (fwp.needSpin){
		eTrunc.Svec.resize(0);
		for (size_t iDir = 0; iDir < 3; iDir++)
			copy(e.S[iDir](b0_dm, b1_dm, b0_dm, b1_dm), eTrunc.S[iDir]);
	}
	if (fwp.needL){
		for (size_t iDir = 0; iDir < 3; iDir++)
			copy(e.L[iDir](b0_dm, b1_dm, b0_dm, b1_dm), eTrunc.L[iDir]);
	}
	if (fwp.needQ){
		for (int iComp = 0; iComp < 5; iComp++)
			copy(e.Q[iComp](b0_dm, b1_dm, b0_dm, b1_dm), eTrunc.Q[iComp]);
	}
	if (fwp.needLayer)
		copy(e.layer(b0_dm, b1_dm, b0_dm, b1_dm), eTrunc.layer);
	if (fwp.needLinewidth_ee)
		eTrunc.E = e.E(b0_probe, b1_probe);
	//do not need linewidths
	//do not save dHePhSum //copy(e.dHePhSum, eTrunc.dHePhSum);
}

void FeynWann::bcastState_JX(FeynWann::StateE& state, MPIUtil* mpiUtil, int root, bool need_dHePhSum)
{	if(mpiUtil->nProcesses()==1) return; //no communictaion needed
	mpiUtil->bcast(state.ik, root);
	mpiUtil->bcast(&state.k[0], 3, root);
	//Energy and eigenvectors:
	bcast(state.E, nBands, mpiUtil, root);
	mpiUtil->bcast(state.withinRange, root);
	if(not state.withinRange) return; //Remaining quantities will never be used
	if (energyOnly) return;
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
	if (fwp.needL){
		for (int iDir = 0; iDir<3; iDir++)
			bcast(state.L[iDir], nBands, nBands, mpiUtil, root);
	}
	//Electric quadrupole r*p matrix, if needed:
	if (fwp.needQ){
		for (int iComp = 0; iComp<3; iComp++)
			bcast(state.Q[iComp], nBands, nBands, mpiUtil, root);
	}
	if (fwp.needLayer)
		bcast(state.layer, nBands, nBands, mpiUtil, root);
	//Linewidths, if needed:
	if (fwp.needLinewidth_ee) bcast(state.ImSigma_ee, nBands, mpiUtil, root);
	if(fwp.needLinewidth_ePh)
	{	state.logImSigma_ePhArr.resize(FeynWannParams::fGrid_ePh.size());
		for (diagMatrix& d : state.logImSigma_ePhArr) bcast(d, nBands, mpiUtil, root);
	}
	if(fwp.needLinewidthP_ePh)
	{	state.logImSigmaP_ePhArr.resize(FeynWannParams::fGrid_ePh.size());
		for (diagMatrix& d : state.logImSigmaP_ePhArr) bcast(d, nBands, mpiUtil, root);
	}
	if (fwp.needLinewidth_D.length()) bcast(state.ImSigma_D, nBands, mpiUtil, root);
	if (fwp.needLinewidthP_D.length()) bcast(state.ImSigmaP_D, nBands, mpiUtil, root);
	//e-ph sum rule if needed
	if (need_dHePhSum)
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
	if (energyOnly) return;
	bcast(state.U, nBands, nBands, mpiUtil, root);
	//Velocity matrix, if needed:
	if (fwp.needVelocity){
		for (int iDir = 0; iDir<3; iDir++)
			bcast(state.v[iDir], nBands, nBands, mpiUtil, root);
		state.vVec.resize(nBands);
		mpiUtil->bcastData(state.vVec, root);
	}
	//Spin matrix, if needed:
	if (fwp.needSpin){
		for (int iDir = 0; iDir<3; iDir++)
			bcast(state.S[iDir], nBands, nBands, mpiUtil, root);
		state.Svec.resize(nBands);
		mpiUtil->bcastData(state.Svec, root);
	}
	//Angular momentum matrix, if needed:
	if (fwp.needL){
		for (int iDir = 0; iDir<3; iDir++)
			bcast(state.L[iDir], nBands, nBands, mpiUtil, root);
	}
	//Electric quadrupole r*p matrix, if needed:
	if (fwp.needQ){
		for (int iComp = 0; iComp<3; iComp++)
			bcast(state.Q[iComp], nBands, nBands, mpiUtil, root);
	}
	if (fwp.needLayer)
		bcast(state.layer, nBands, nBands, mpiUtil, root);
	//Linewidths, if needed:
	if (fwp.needLinewidth_ee) bcast(state.ImSigma_ee, nBands, mpiUtil, root);
	if (fwp.needLinewidth_ePh){
		state.logImSigma_ePhArr.resize(FeynWannParams::fGrid_ePh.size());
		for (diagMatrix& d : state.logImSigma_ePhArr) bcast(d, nBands, mpiUtil, root);
	}
	if (fwp.needLinewidthP_ePh){
		state.logImSigmaP_ePhArr.resize(FeynWannParams::fGrid_ePh.size());
		for (diagMatrix& d : state.logImSigmaP_ePhArr) bcast(d, nBands, mpiUtil, root);
	}
	//e-ph sum rule if needed
	bcast(state.dHePhSum, nBands*nBands, 3, mpiUtil, root);
}