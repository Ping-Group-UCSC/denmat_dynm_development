#include "help_lindbladInit_for-DMD.h"

int dimension(FeynWann& fw){
	double d = 3;
	for (int iDir = 0; iDir<3; iDir++)
	if (fw.isTruncated[iDir]) d--;
	return d;
}
double cell_size(FeynWann& fw){
	if (dimension(fw) == 3) return fabs(det(fw.R));
	else if (dimension(fw) == 2) return fabs(det(fw.R)) / fw.R(2, 2);
	else if (dimension(fw) == 1) return fw.R(2, 2);
	else return 1;
}
double cminvdim2au(FeynWann& fw){
	return std::pow(bohr2cm, dimension(fw));
}
void print_carrier_density(FeynWann& fw, double carrier_density){
	carrier_density /= cminvdim2au(fw);
	if (dimension(fw) == 3)
		logPrintf("Carrier density: %14.7le cm^-3\n", carrier_density);
	else if (dimension(fw) == 2)
		logPrintf("Carrier density: %14.7le cm^-2\n", carrier_density);
	else if (dimension(fw) == 1)
		logPrintf("Carrier density: %14.7le cm^-1\n", carrier_density);
	else
		logPrintf("Carrier density: %14.7le\n", carrier_density);
}

double find_mu(double ncarrier, double t, double mu0, std::vector<FeynWann::StateE>& e, int bStart, int bCBM, int bStop){
	// notice that ncarrier must be carrier density * cell_size * nkTot
	double result = mu0;
	double damp = 0.7, dmu = 5e-6;
	double excess_old_old = 1., excess_old = 1., excess, ncarrier_new;
	int step = 0;

	std::vector<diagMatrix> Ek(e.size());
	for (size_t ik = 0; ik < e.size(); ik++)
		Ek[ik] = e[ik].E(bStart, bStop);

	while (true){
		ncarrier_new = compute_ncarrier(ncarrier < 0, t, result, Ek, bStop - bStart, bCBM - bStart);
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
				ncarrier_new = compute_ncarrier(ncarrier < 0, t, result, Ek, bStop - bStart, bCBM - bStart);
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
	logPrintf("\nmu0 = %14.7le mu = %14.7le:\n", mu0, result);
	logPrintf("Carriers per cell = %lg excess = %lg\n", ncarrier, excess);
	//}
	return result;
}
double compute_ncarrier(bool isHole, double t, double mu, std::vector<diagMatrix>& Ek, int nb, int nv){
	size_t ikStart, ikStop; //range of offstes handled by this process group
	if (mpiGroup->isHead()) TaskDivision(Ek.size(), mpiGroupHead).myRange(ikStart, ikStop);
	mpiGroup->bcast(ikStart); mpiGroup->bcast(ikStop);
	MPI_Barrier(MPI_COMM_WORLD);

	double result = 0.;
	for (size_t ik = ikStart; ik < ikStop; ik++)
	for (int i = 0; i < nb; i++){
		double f = fermi((Ek[ik][i] - mu) / t);
		if (isHole && i < nv) result += f - 1; // hole concentration is negative
		if (!isHole && i >= nv) result += f;
	}

	mpiWorld->allReduce(result, MPIUtil::ReduceSum);
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

vector3<> compute_bsq(std::vector<FeynWann::StateE>& e, int bStart, int bStop, double degthr, std::vector<diagMatrix> F){
	int nb = bStop - bStart;
	vector3<> bsq_avg(vector3<>(0,0,0));
	double sum = 0;
	for (size_t ik = 0; ik < e.size(); ik++){
		diagMatrix Ek = e[ik].E(bStart, bStop), dfde(nb);
		for (int b = 0; b < nb; b++){
			dfde[b] = F[ik][b] * (1 - F[ik][b]);
			sum += dfde[b];
		}
		for (int id = 0; id < 3; id++){
			matrix s = e[ik].S[id](bStart, bStop, bStart, bStop);
			matrix sdeg = degProj(s, Ek, degthr);
			diagMatrix ss = diag(sdeg*sdeg);
			// a^2 + b^2 = 1, a^2 - b^2 = sdeg^2 => b^2 = (1 - sdeg^2) / 2
			for (int b = 0; b < nb; b++){
				double bsq = 0.5 * (1 - sqrt(ss[b]));
				bsq_avg[id] += dfde[b] * bsq;
			}
		}
	}
	return bsq_avg / sum;
}
matrix degProj(matrix& M, diagMatrix& E, double degthr){
	matrix Mdeg(E.size(), E.size());
	complex *MdegData = Mdeg.data();
	for (int b2 = 0; b2 < E.size(); b2++)
	for (int b1 = 0; b1 < E.size(); b1++){
		if (fabs(E[b1] - E[b2]) >= degthr) *MdegData = c0;
		else *MdegData = M(b1, b2);
		MdegData++;
	}
	return Mdeg;
}
void degProj(matrix& M, diagMatrix& E, double degthr, matrix& Mdeg){
	complex *MdegData = Mdeg.data();
	for (int b2 = 0; b2 < E.size(); b2++)
	for (int b1 = 0; b1 < E.size(); b1++){
		if (fabs(E[b1] - E[b2]) >= degthr) *MdegData = c0;
		else *MdegData = M(b1, b2);
		MdegData++;
	}
}
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
void init_dm(complex **dm, size_t nk, int nb, std::vector<diagMatrix>& F){
	for (size_t ik = 0; ik < nk; ik++)
	for (int b1 = 0; b1 < nb; b1++)
	for (int b2 = 0; b2 < nb; b2++)
	if (b1 == b2)
		dm[ik][b1*nb + b2] = F[ik][b1];
	else
		dm[ik][b1*nb + b2] = c0;
}
void set_dm1(complex **dm, size_t nk, int nb, complex **dm1){
	for (size_t ik = 0; ik < nk; ik++)
	for (int b1 = 0; b1 < nb; b1++)
	for (int b2 = 0; b2 < nb; b2++)
	if (b1 == b2)
		dm1[ik][b1*nb + b2] = c1 - dm[ik][b1*nb + b2];
	else
		dm1[ik][b1*nb + b2] = -dm[ik][b1*nb + b2];
}

double** alloc_real_array(int n1, int n2, double val){
	double** ptr = nullptr;
	double* pool = nullptr;
	if (n1 * n2 == 0) return ptr;
	try{
		ptr = new double*[n1];  // allocate pointers (can throw here)
		pool = new double[n1*n2]{val};  // allocate pool (can throw here)
		for (int i = 0; i < n1; i++, pool += n2)
			ptr[i] = pool; // now point the row pointers to the appropriate positions in the memory pool
		return ptr;
	}
	catch (std::bad_alloc& ex){ delete[] ptr; throw ex; }
}
complex* alloc_array(int n1, complex val){
	complex* arr;
	if (n1 == 0) return arr;
	try{ arr = new complex[n1]{val}; }
	catch (std::bad_alloc& ba){ printf("bad_alloc of arr caught: %s in alloc_array(n1)\n", ba.what()); }
	return arr;
}
complex** alloc_array(int n1, int n2, complex val){
	complex** ptr = nullptr;
	complex* pool = nullptr;
	if (n1 * n2 == 0) return ptr;
	try{
		ptr = new complex*[n1];  // allocate pointers (can throw here)
		pool = new complex[n1*n2]{val};  // allocate pool (can throw here)
		for (int i = 0; i < n1; i++, pool += n2)
			ptr[i] = pool; // now point the row pointers to the appropriate positions in the memory pool
		return ptr;
	}
	catch (std::bad_alloc& ex){ delete[] ptr; throw ex; }
}
complex*** alloc_array(int n1, int n2, int n3, complex val){
	complex*** arr;
	if (n1 == 0) return arr;
	try{ arr = new complex**[n1]; }
	catch (std::bad_alloc& ba){ printf("bad_alloc of arr caught: %s in alloc_array(n1,n2,n3)\n", ba.what()); }
	for (int i1 = 0; i1 < n1; i1++)
		arr[i1] = alloc_array(n2, n3, val);
	return arr;
}

void zeros(double* arr, int n1){
	for (int i1 = 0; i1 < n1; i1++)
		arr[i1] = 0.;
}
void zeros(double** arr, int n1, int n2){
	for (int i1 = 0; i1 < n1; i1++)
		zeros(arr[i1], n2);
}
void zeros(complex* arr, int n1){
	for (int i1 = 0; i1 < n1; i1++)
		arr[i1] = c0;
}
void zeros(complex** arr, int n1, int n2){
	for (int i1 = 0; i1 < n1; i1++)
		zeros(arr[i1], n2);
}
void zeros(complex*** arr, int n1, int n2, int n3){
	for (int i1 = 0; i1 < n1; i1++)
		zeros(arr[i1], n2, n3);
}
void zeros(matrix& m){
	complex *mData = m.data();
	for (int i = 0; i < m.nRows()*m.nCols(); i++)
		*(mData++) = complex(0, 0);
}
void zeros(std::vector<matrix>& v){
	for (matrix& m : v)
		zeros(m);
}

void axbyc(complex *y, complex *x, int n, complex a, complex b, complex c){
	if (b.real() == 0 && b.imag() == 0) zeros(y, n);
	else if (!(b.real() == 1 && b.imag() == 0))
	for (int i = 0; i < n; i++){ y[i] *= b; }
	if (x == nullptr || (a.real() == 0 && a.imag() == 0))
	for (int i = 0; i < n; i++){ y[i] += c; }
	else if (a.real() == 1 && a.imag() == 0)
	for (int i = 0; i < n; i++){ y[i] += x[i] + c; }
	else
	for (int i = 0; i < n; i++){ y[i] += a * x[i] + c; }
}

void error_message(string s, string routine){
	printf((s + " in " + routine).c_str());
	exit(EXIT_FAILURE);
}
void printf_complex_mat(complex *m, int n, string s){
	if (n < 3){
		printf("%s", s.c_str());
		for (int i = 0; i < n*n; i++)
			printf(" (%lg,%lg)", m[i].real(), m[i].imag());
		printf("\n");
	}
	else{
		printf("%s\n", s.c_str());
		for (int i = 0; i < n; i++){
			for (int j = 0; j < n; j++)
				printf(" (%lg,%lg)", m[i*n + j].real(), m[i*n + j].imag());
			printf("\n");
		}
	}
}
void fprintf_complex_mat(FILE *fp, complex *m, int n, string s){
	if (n < 3){
		fprintf(fp, "%s", s.c_str());
		for (int i = 0; i < n*n; i++)
			fprintf(fp, " (%lg,%lg)", m[i].real(), m[i].imag());
		fprintf(fp, "\n");
	}
	else{
		fprintf(fp, "%s\n", s.c_str());
		for (int i = 0; i < n; i++){
			for (int j = 0; j < n; j++)
				fprintf(fp, " (%lg,%lg)", m[i*n + j].real(), m[i*n + j].imag());
			fprintf(fp, "\n");
		}
	}
}

void check_file_size(FILE *fp, size_t expect_size, string message){
	fseek(fp, 0L, SEEK_END);
	size_t sz = ftell(fp);
	if (sz != expect_size){
		printf("file size is %lu while expected size is %lu", sz, expect_size);
		error_message(message);
	}
	rewind(fp);
}
void fseek_bigfile(FILE *fp, size_t count, size_t size, int origin){
	size_t count_step = 100000000 / size;
	size_t nsteps = count / count_step;
	size_t reminder = count % count_step;
	size_t pos = reminder * size;
	size_t accum = reminder;
	fseek(fp, pos, origin);
	for (size_t istep = 0; istep < nsteps; istep++){
		pos = count_step * size;
		fseek(fp, pos, SEEK_CUR);
		accum += count_step;
	}
	if (accum != count)
		error_message("accum != count", "fseek_bigfile");
}
