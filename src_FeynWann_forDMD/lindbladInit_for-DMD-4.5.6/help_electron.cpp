#include "help_electron.h"

double compute_sz(complex **dm, size_t nk, double nkTot, int nb, int bStart, int bStop, std::vector<FeynWann::StateE>& e){
	double result = 0.;
	for (size_t ik = 0; ik < nk; ik++){
		matrix s = e[ik].S[2](bStart, bStop, bStart, bStop);
		for (int b2 = 0; b2 < nb; b2++)
		for (int b1 = 0; b1 < nb; b1++)
			result += real(s(b1, b2) * dm[ik][b2*nb + b1]);
	}
	return result / nkTot;
}
vector3<> compute_spin(std::vector<std::vector<matrix>> m, size_t nk, double nkTot, int nb, int bStart, int bStop, std::vector<FeynWann::StateE>& e){
	vector3<> result(0., 0., 0.);
	for (size_t ik = 0; ik < nk; ik++)
	for (int id = 0; id < 3; id++){
		matrix s = e[ik].S[id](bStart, bStop, bStart, bStop);
		for (int b2 = 0; b2 < nb; b2++)
		for (int b1 = 0; b1 < nb; b1++)
			result[id] += real(s(b1, b2) * m[ik][id](b2, b1));
	}
	return result / nkTot;
}

double maxval(std::vector<FeynWann::StateE>& e, int bStart, int bStop){
	double r = DBL_MIN;
	for (size_t ik; ik < e.size(); ik++){
		diagMatrix Ek = e[ik].E(bStart, bStop);
		for (int b = 0; b < bStop - bStart; b++)
		if (Ek[b] > r) r = Ek[b];
	}
	return r;
}
double minval(std::vector<FeynWann::StateE>& e, int bStart, int bStop){
	double r = DBL_MAX;
	for (size_t ik; ik < e.size(); ik++){
		diagMatrix Ek = e[ik].E(bStart, bStop);
		for (int b = 0; b < bStop - bStart; b++)
		if (Ek[b] < r) r = Ek[b];
	}
	return r;
}

double find_mu(double ncarrier, double t, double mu0, std::vector<double>& E, int nk, int bStart, int bCBM, int bStop){
	int nb_E = E.size() / nk;
	std::vector<diagMatrix> Ek(nk, diagMatrix(bStop - bStart));
	for (size_t ik = 0; ik < nk; ik++)
	for (size_t i = 0; i < bStop - bStart; i++)
		Ek[ik][i] = E[ik*nb_E + i + bStart];

	return find_mu(ncarrier, t, mu0, Ek, bCBM - bStart);
}
double find_mu(double ncarrier, double t, double mu0, std::vector<FeynWann::StateE>& e, int bStart, int bCBM, int bStop){
	std::vector<diagMatrix> Ek(e.size());
	for (size_t ik = 0; ik < e.size(); ik++)
		Ek[ik] = e[ik].E(bStart, bStop);

	return find_mu(ncarrier, t, mu0, Ek, bCBM - bStart);
}
double find_mu(double ncarrier, double t, double mu0, std::vector<diagMatrix>& Ek, int nv){
	// notice that ncarrier must be carrier density * cell_size * nkTot
	double result = mu0;
	double damp = 0.7, dmu = 5e-6;
	double excess_old_old = 1., excess_old = 1., excess, ncarrier_new;
	int step = 0;

	while (true){
		ncarrier_new = compute_ncarrier(ncarrier < 0, t, result, Ek, nv);
		excess = ncarrier_new - ncarrier;
		if (fabs(excess) > 1e-14){
			if (fabs(excess) > fabs(excess_old) || fabs(excess) > fabs(excess_old_old))
				dmu *= damp;
			result -= sgn(excess) * dmu;

			// the shift of mu should be large when current mu is far from converged one
			if (step > 0 && sgn(excess) == sgn(excess_old)){
				double ratio = ncarrier_new / ncarrier;
				if (ratio < 1e-9)
					result -= sgn(excess) * 10 * t;
				else if (ratio < 1e-4)
					result -= sgn(excess) * 3 * t;
				else if (ratio < 0.1)
					result -= sgn(excess) * 0.7 * t;
			}

			if (dmu < 1e-16){
				ncarrier_new = compute_ncarrier(ncarrier < 0, t, result, Ek, nv);
				excess = ncarrier_new - ncarrier;
				break;
			}

			excess_old_old = excess_old;
			excess_old = excess;
			step++;
			if (step > 1e3) break;
		}
		else
			break;
	}

	//if ((fabs(t) < 1e-6 || fabs(excess) > 1e-10)){
	logPrintf("\nmu0 = %14.7le Ha (%14.7le eV) mu = %14.7le Ha (%14.7le eV):\n", mu0, mu0/eV, result, result/eV);
	logPrintf("Carriers per cell = %lg excess = %lg\n", ncarrier, excess);
	//}
	return result;
}
double compute_ncarrier(bool isHole, double t, double mu, std::vector<diagMatrix>& Ek, int nv){
	size_t ikStart, ikStop; //range of offstes handled by this process group
	if (mpiGroup->isHead()) TaskDivision(Ek.size(), mpiGroupHead).myRange(ikStart, ikStop);
	mpiGroup->bcast(ikStart); mpiGroup->bcast(ikStop);
	MPI_Barrier(MPI_COMM_WORLD);

	double result = 0.;
	for (size_t ik = ikStart; ik < ikStop; ik++)
	for (size_t i = 0; i < Ek[0].size(); i++){
		double f = fermi((Ek[ik][i] - mu) / t);
		if (isHole && i < nv) result += (f - 1); // hole concentration is negative
		if (!isHole && i >= nv) result += f;
	}

	mpiGroupHead->allReduce(result, MPIUtil::ReduceSum);
	mpiGroup->bcast(result);
	return result;
}
double compute_ncarrier(bool isHole, double t, double mu, std::vector<FeynWann::StateE>& e, int bStart, int bCBM, int bStop){
	size_t ikStart, ikStop; //range of offstes handled by this process group
	if (mpiGroup->isHead()) TaskDivision(e.size(), mpiGroupHead).myRange(ikStart, ikStop);
	mpiGroup->bcast(ikStart); mpiGroup->bcast(ikStop);
	MPI_Barrier(MPI_COMM_WORLD);

	double result = 0.;
	for (size_t ik = ikStart; ik < ikStop; ik++)
	for (int b = bStart; b < bStop; b++){
		double f = fermi((e[ik].E[b] - mu) / t);
		if (isHole && b < bCBM) result += f - 1; // hole concentration is negative
		if (!isHole && b >= bCBM) result += f;
	}

	mpiWorld->allReduce(result, MPIUtil::ReduceSum);
	return result;
}
std::vector<diagMatrix> computeF(double t, double mu, std::vector<FeynWann::StateE>& e, int bStart, int bStop){
	std::vector<diagMatrix> F(e.size(), diagMatrix(bStop - bStart));
	for (size_t ik = 0; ik < e.size(); ik++){
		diagMatrix Ek = e[ik].E(bStart, bStop);
		for (int b = 0; b < bStop - bStart; b++){
			F[ik][b] = fermi((Ek[b] - mu) / t);
		}
	}
	return F;
}