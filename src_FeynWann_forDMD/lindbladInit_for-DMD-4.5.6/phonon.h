#pragma once
#include "common_headers.h"
#include "parameters.h"
#include "electron.h"

class phonon{
public:
	FeynWann& fw;
	double omegaMax;
	std::vector<vector3<>> q0;

	phonon(FeynWann& fw, parameters *param) : fw(fw)
	{
		param->modeStop = param->modeStop < 0 ? fw.nModes : param->modeStop; assert(param->modeStop <= fw.nModes);
		logPrintf("modeStop = %d\n", param->modeStop);
		if (param->needOmegaPhMax) findMaxOmega();
		if (param->ePhEnabled) get_q_offsets(param->NkMult);
	}

	void get_q_offsets(vector3<int> NkMult){
		vector3<int> NkFine;
		for (int iDir = 0; iDir<3; iDir++){
			if (fw.isTruncated[iDir] && NkMult[iDir] != 1){
				logPrintf("Setting NkMult = 1 along truncated direction %d.\n", iDir + 1);
				NkMult[iDir] = 1; //no multiplication in truncated directions
			}
			NkFine[iDir] = fw.kfold[iDir] * NkMult[iDir];
		}
		matrix3<> NkFineInv = inv(Diag(vector3<>(NkFine)));

		vector3<int> NqMult;
		for (int iDir = 0; iDir<3; iDir++)
			NqMult[iDir] = NkFine[iDir] / fw.phononSup[iDir];
		vector3<int> iqMult;
		for (iqMult[0] = 0; iqMult[0]<NqMult[0]; iqMult[0]++)
		for (iqMult[1] = 0; iqMult[1]<NqMult[1]; iqMult[1]++)
		for (iqMult[2] = 0; iqMult[2]<NqMult[2]; iqMult[2]++)
			q0.push_back(NkFineInv * iqMult);
	}

	inline void findMaxOmega(const FeynWann::StatePh& state){
		omegaMax = std::max(omegaMax, state.omega.back()); //omega is in ascending order
	}
	static void findMaxOmega(const FeynWann::StatePh& state, void* params){
		((phonon*)params)->findMaxOmega(state);
	}
	void findMaxOmega(){
		// find maximum phonon energy
		omegaMax = 0;
		fw.phLoop(vector3<>(), findMaxOmega, this);
		mpiWorld->allReduce(omegaMax, MPIUtil::ReduceMax);
		omegaMax *= 1.1; //add some margin
		logPrintf("Maximum phonon energy: %lg meV\n", omegaMax / eV * 1000);
	}
};