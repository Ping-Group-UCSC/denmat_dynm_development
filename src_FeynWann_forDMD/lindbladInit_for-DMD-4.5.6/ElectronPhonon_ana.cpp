#include "ElectronPhonon.h"

void ElectronPhonon::compute_eph_analyse_g2_E1E2fix(){
	double sum_g2nq = 0, sum_sg2nq = 0, sum_weight = 0;
	double prefac_exp = -0.5 / std::pow(ePhDelta, 2);

	std::vector<std::vector<matrix>> Sdeg(k.size(), std::vector<matrix>(3, matrix(nBandsSel, nBandsSel)));
	for (size_t ik = 0; ik < k.size(); ik++){
		//diagMatrix Ek = elec->state_elec[ik].E(bStart, bStop);
		diagMatrix Ek = elec->state_elec[ik].E(bStart - elec->bRef, bStop - elec->bRef);
		for (int id = 0; id < 3; id++){
			//matrix Sfull = elec->state_elec[ik].S[id](bStart, bStop, bStart, bStop);
			matrix Sfull = elec->state_elec[ik].S[id](bStart - elec->bBot_dm, bStop - elec->bBot_dm, bStart - elec->bBot_dm, bStop - elec->bBot_dm);
			degProj(Sfull, Ek, degthr, Sdeg[ik][id]);
		}
	}

	TaskDivision(kpairs.size(), mpiWorld).myRange(ikpairStart, ikpairStop);
	nkpairMine = ikpairStop - ikpairStart;
	size_t nkpairInterval = std::max(1, int(round(nkpairMine / 50.))); //interval for reporting progress
	MPI_Barrier(MPI_COMM_WORLD);

	logPrintf("Compute EPH for analyse_g2_E1E2fix: \n"); logFlush();
	for (int ikpair_local = 0; ikpair_local < nkpairMine; ikpair_local++){
		int ikpair_glob = ikpair_local + ikpairStart;
		int ik = kpairs[ikpair_glob].first, jk = kpairs[ikpair_glob].second;
		if (onlyInterValley && !latt->isInterValley(k[ik], k[jk])) continue;
		if (onlyIntraValley && latt->isInterValley(k[ik], k[jk])) continue;
		//diagMatrix Ek = elec->state_elec[ik].E(bStart, bStop), Ekp = elec->state_elec[jk].E(bStart, bStop);
		diagMatrix Ek = elec->state_elec[ik].E(bStart - elec->bRef, bStop - elec->bRef), Ekp = elec->state_elec[jk].E(bStart - elec->bRef, bStop - elec->bRef);

		FeynWann::StatePh phm, phm_eph;
		if (modeStop > modeStart) fw.phCalc(k[ik] - k[jk], phm); // q = k - k'
		FeynWann::MatrixEph mm;
		if (modeStop > modeStart) fw.ePhCalc(elec->state_elec[ik], elec->state_elec[jk], phm, mm); // g^-_kk'

		matrix S1z = Sdeg[ik][2], S2z = Sdeg[jk][2];

		for (int b1 = 0; b1 < nBandsSel; b1++){
			double w1 = exp(prefac_exp * std::pow(Ek[b1] - E1fix, 2));

			for (int b2 = 0; b2 < nBandsSel; b2++){
				double w2 = exp(prefac_exp * std::pow(Ekp[b2] - E2fix, 2));
				double weight = w1*w2;
				sum_weight += weight;

				double sum1 = 0, sum2 = 0;
				for (int alpha = modeStart; alpha < modeStop; alpha++){
					//matrix gm = mm.M[alpha](bStart, bStop, bStart, bStop);
					matrix gm = mm.M[alpha];

					double nq = bose(phm.omega[alpha] / Tmax);
					matrix SGcomm = S1z * gm - gm * S2z;

					sum1 += gm(b1, b2).norm() * nq;
					sum2 += SGcomm(b1, b2).norm() * nq;
				}

				sum_g2nq += weight *sum1;
				sum_sg2nq += weight * sum2;
			}
		}

		//Print progress:
		if ((ikpair_local + 1) % nkpairInterval == 0) { logPrintf("%d%% ", int(round((ikpair_local + 1)*100. / nkpairMine))); logFlush(); }
	}
	MPI_Barrier(MPI_COMM_WORLD);
	logPrintf("done.\n"); logFlush();

	mpiWorld->reduce(sum_weight, MPIUtil::ReduceSum);
	mpiWorld->reduce(sum_g2nq, MPIUtil::ReduceSum);
	mpiWorld->reduce(sum_sg2nq, MPIUtil::ReduceSum);

	logPrintf("Average |g|^2*nq = %lg\n", sum_g2nq / sum_weight);
	logPrintf("Average |[s,g]|^2*nq = %lg\n", sum_sg2nq / sum_weight);
	logPrintf("sum of weights = %lg (%lu)\n", sum_weight, kpairs.size());
}