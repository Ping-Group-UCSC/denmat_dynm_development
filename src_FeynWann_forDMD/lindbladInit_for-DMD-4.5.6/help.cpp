#include "help.h"

vector3<> wrap_around_Gamma(const vector3<>& x){
	vector3<> result = x;
	for (int dir = 0; dir<3; dir++)
		result[dir] -= floor(0.5 + result[dir]);
	return result;
}
vector3<> wrap(const vector3<>& x, vector3<> center){
	// wrap to [center - 0.5, center + 0.5)
	vector3<> result = x - center;
	for (int dir = 0; dir < 3; dir++){
		result[dir] -= floor(0.5 + result[dir]);
		if (fabs(result[dir] - 0.5) < 1e-6) result[dir] = -0.5;
	}
	result = result + center;
	return result;
}

matrix degProj(matrix& M, diagMatrix& E, double degthr){
	matrix Mdeg(E.size(), E.size());
	complex *MdegData = Mdeg.data();
	for (int b2 = 0; b2 < (int)E.size(); b2++)
	for (int b1 = 0; b1 < (int)E.size(); b1++){
		if (fabs(E[b1] - E[b2]) >= degthr) *MdegData = c0;
		else *MdegData = M(b1, b2);
		MdegData++;
	}
	return Mdeg;
}
void degProj(matrix& M, diagMatrix& E, double degthr, matrix& Mdeg){
	complex *MdegData = Mdeg.data();
	for (int b2 = 0; b2 < (int)E.size(); b2++)
	for (int b1 = 0; b1 < (int)E.size(); b1++){
		if (fabs(E[b1] - E[b2]) >= degthr) *MdegData = c0;
		else *MdegData = M(b1, b2);
		MdegData++;
	}
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

void axbyc(double *y, double *x, size_t n, double a, double b, double c){
	if (b == 0) zeros(y, n);
	else if (b != 1) for (size_t i = 0; i < n; i++) { y[i] *= b; }
	if (x == nullptr || a == 0) for (size_t i = 0; i < n; i++){ y[i] += c; }
	else if (a == 1) for (size_t i = 0; i < n; i++) { y[i] += x[i] + c; }
	else for (size_t i = 0; i < n; i++) { y[i] += a * x[i] + c; }
}
void axbyc(double *y, double *x, int n, double a, double b, double c){
	axbyc(y, x, (size_t)n, a, b, c);
}
void axbyc(double **y, double **x, int n1, int n2, double a, double b, double c){
	if (b == 0)
		zeros(y, n1, n2);
	else if (b != 1){
		for (int i1 = 0; i1 < n1; i1++)
		for (int i2 = 0; i2 < n2; i2++)
			y[i1][i2] *= b;
	}
	if (x == nullptr || a == 0){
		for (int i1 = 0; i1 < n1; i1++)
		for (int i2 = 0; i2 < n2; i2++)
			y[i1][i2] += c;
	}
	else if (a == 1){
		for (int i1 = 0; i1 < n1; i1++)
		for (int i2 = 0; i2 < n2; i2++)
			y[i1][i2] += x[i1][i2] + c;
	}
	else{
		for (int i1 = 0; i1 < n1; i1++)
		for (int i2 = 0; i2 < n2; i2++)
			y[i1][i2] += a * x[i1][i2] + c;
	}
}
void axbyc(complex **y, complex **x, int n1, int n2, complex a, complex b, complex c){
	if (b.real() == 0 && b.imag() == 0)
		zeros(y, n1, n2);
	else if (!(b.real() == 1 && b.imag() == 0)){
		for (int i1 = 0; i1 < n1; i1++)
		for (int i2 = 0; i2 < n2; i2++)
			y[i1][i2] *= b;
	}
	if (x == nullptr || (a.real() == 0 && a.imag() == 0)){
		for (int i1 = 0; i1 < n1; i1++)
		for (int i2 = 0; i2 < n2; i2++)
			y[i1][i2] += c;
	}
	else if (a.real() == 1 && a.imag() == 0){
		for (int i1 = 0; i1 < n1; i1++)
		for (int i2 = 0; i2 < n2; i2++)
			y[i1][i2] += x[i1][i2] + c;
	}
	else{
		for (int i1 = 0; i1 < n1; i1++)
		for (int i2 = 0; i2 < n2; i2++)
			y[i1][i2] += a * x[i1][i2] + c;
	}
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

double mean_of_array(double *a, int n, double *w){
	if (n == 0) return 0;
	double sum = 0;
	if (w == nullptr){
		for (int i = 0; i < n; i++)
			sum += a[i];
		return sum / n;
	}
	else{
		double sumw = 0;
		for (int i = 0; i < n; i++){
			sum += a[i] * w[i];
			sumw += w[i];
		}
		return sum / sumw;
	}
}
double sigma_of_array(double *a, int n, bool compute_mean, double mean_fixed, double *w){
	if (n <= 1) return 0;
	double sum = 0, mean1;
	if (compute_mean) mean1 = mean_of_array(a, n, w);
	else mean1 = mean_fixed;
	if (w == nullptr){
		for (int i = 0; i < n; i++)
			sum += std::pow(a[i] - mean1, 2.);
		return sqrt(sum / n);
	}
	else{
		double sumw = 0;
		for (int i = 0; i < n; i++){
			sum += std::pow(a[i] - mean1, 2.) * w[i];
			sumw += w[i];
		}
		return sqrt(sum / sumw);
	}
}
void random_normal_array(double *a, int n, double mean, double sigma, double cap, double *w){
	if (n == 0) return;
	if (n == 1) { a[0] = mean; return; }
	Random::seed(n);
	while (true){
		for (int i = 0; i < n; i++)
			a[i] = Random::normal(mean, sigma, cap);
		if (sigma == 0) return;
		// check mean of obtained array and shift to target mean
		double mean1 = mean_of_array(a, n, w);
		for (int i = 0; i < n; i++)
			a[i] = a[i] + mean - mean1;
		// check sigma of array a and scale it to get target sigma
		double sigma1 = sigma_of_array(a, n, false, mean, w);
		if (sigma1 == 0) continue; // ensure sigma1 is not zero
		double fac = sigma / sigma1;
		//double fac = sqrt(sigma / sigma1);
		for (int i = 0; i < n; i++){
			double d = a[i] - mean;
			a[i] = mean + fac * d;
		}
		return;
	}
}
vector3<> mean_of_(std::vector<vector3<>> m, double *w){
	vector3<> result;
	std::vector<double> v(m.size());
	for (int id = 0; id < 3; id++){
		for (size_t ik = 0; ik < m.size(); ik++)
			v[ik] = m[ik][id];
		result[id] = mean_of_array(v.data(), v.size(), w);
	}
	return result;
}
vector3<> sigma_of_(std::vector<vector3<>> m, bool compute_mean, vector3<> mean_fixed, double *w){
	vector3<> result;
	std::vector<double> v(m.size());
	for (int id = 0; id < 3; id++){
		for (size_t ik = 0; ik < m.size(); ik++)
			v[ik] = m[ik][id];
		result[id] = sigma_of_array(v.data(), v.size(), compute_mean, mean_fixed[id], w);
	}
	return result;
}
void random_normal_(std::vector<vector3<>>& m, vector3<> mean, vector3<> sigma, vector3<> cap, double *w){
	std::vector<double> v(m.size());
	for (int id = 0; id < 3; id++){
		random_normal_array(v.data(), v.size(), mean[id], sigma[id], cap[id], w);
		for (size_t ik = 0; ik < m.size(); ik++)
			m[ik][id] = v[ik];
	}
}
matrix3<> mean_of_(std::vector<matrix3<>> m, double *w){
	matrix3<> result;
	std::vector<double> v(m.size());
	for (int id = 0; id < 3; id++)
	for (int jd = 0; jd < 3; jd++){
		for (size_t ik = 0; ik < m.size(); ik++)
			v[ik] = m[ik](id, jd);
		result(id, jd) = mean_of_array(v.data(), v.size(), w);
	}
	return result;
}
matrix3<> sigma_of_(std::vector<matrix3<>> m, bool compute_mean, matrix3<> mean_fixed, double *w){
	matrix3<> result;
	std::vector<double> v(m.size());
	for (int id = 0; id < 3; id++)
	for (int jd = 0; jd < 3; jd++){
		for (size_t ik = 0; ik < m.size(); ik++)
			v[ik] = m[ik](id, jd);
		result(id, jd) = sigma_of_array(v.data(), v.size(), compute_mean, mean_fixed(id, jd), w);
	}
	return result;
}
void random_normal_(std::vector<matrix3<>>& m, matrix3<> mean, matrix3<> sigma, matrix3<> cap, double *w){
	std::vector<double> v(m.size());
	for (int id = 0; id < 3; id++)
	for (int jd = 0; jd < 3; jd++){
		random_normal_array(v.data(), v.size(), mean(id, jd), sigma(id, jd), cap(id, jd), w);
		for (size_t ik = 0; ik < m.size(); ik++)
			m[ik](id, jd) = v[ik];
	}
}

void fprintf_real_array(FILE *fp, double *a, int n, string s, bool col_mode){
	fprintf_real_array(fp, a, n, s, "%lg", col_mode);
}
void fprintf_real_array(string fname, double *a, int n, string s, string fmt_val, bool col_mode){
	FILE *fp = fopen(fname.c_str(), "w");
	fprintf_real_array(fp, a, n, s, fmt_val, col_mode);
	fclose(fp);
}
void fprintf_real_array(string fname, double *a, int n, string s, bool col_mode){
	FILE *fp = fopen(fname.c_str(), "w");
	fprintf_real_array(fp, a, n, s, col_mode);
	fclose(fp);
}
void fprintf_real_array(FILE *fp, double *a, int n, string s, string fmt_val, bool col_mode){
	if (col_mode){
		fprintf(fp, "%s\n", s.c_str());
		string fmt = fmt_val + "\n";
		for (int i = 0; i < n; i++)
			fprintf(fp, fmt.c_str(), a[i]);
	}
	else{
		fprintf(fp, "%s", s.c_str());
		string fmt = " " + fmt_val;
		for (int i = 0; i < n; i++)
			fprintf(fp, fmt.c_str(), a[i]);
	}
	fprintf(fp, "\n");
}
void fprintf(string fname, std::vector<vector3<>> m, string s, string fmt_val){
	FILE *fp = fopen(fname.c_str(), "w");
	fprintf(fp, m, s, fmt_val);
	fclose(fp);
}
void fprintf(FILE *fp, std::vector<vector3<>> m, string s, string fmt_val){
	fprintf(fp, "%s\n", s.c_str());
	for (size_t i = 0; i < m.size(); i++){
		for (int j = 0; j < 3; j++)
			fprintf(fp, fmt_val.c_str(), m[i][j]);
		fprintf(fp, "\n");
	}
}
void fprintf(string fname, std::vector<matrix3<>> m, string s, string fmt_val){
	FILE *fp = fopen(fname.c_str(), "w");
	fprintf(fp, m, s, fmt_val);
	fclose(fp);
}
void fprintf(FILE *fp, std::vector<matrix3<>> m, string s, string fmt_val){
	fprintf(fp, "%s\n", s.c_str());
	for (size_t i = 0; i < m.size(); i++){
		for (int j = 0; j < 3; j++)
		for (int k = 0; k < 3; k++)
			fprintf(fp, fmt_val.c_str(), m[i](j, k));
		fprintf(fp, "\n");
	}
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

bool exists(string name){
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}
string int2str(int i){
	ostringstream oss; oss << i;
	return oss.str();
}
size_t file_size(FILE *fp){
	fseek(fp, 0L, SEEK_END);
	size_t sz = ftell(fp);
	rewind(fp);
	return sz;
}
void check_file_size(FILE *fp, size_t expect_size, string message){
	size_t sz = file_size(fp);
	if (sz != expect_size){
		printf("file size is %lu while expected size is %lu", sz, expect_size);
		error_message(message);
	}
}
bool check_file_size(FILE *fp, size_t expect_size){
	size_t sz = file_size(fp);
	if (sz != expect_size) return false;
	else true;
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
