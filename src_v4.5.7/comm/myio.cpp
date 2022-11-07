#include <myio.h>

void printf_real_array(double *a, int n, string s, bool col_mode){
	fprintf_real_array(stdout, a, n, s, col_mode);
}
void fprintf_real_array(string fname, double *a, int n, string s, bool col_mode){
	FILE *fp = fopen(fname.c_str(), "w");
	fprintf_real_array(fp, a, n, s, col_mode);
	fclose(fp);
}
void fprintf_real_array(FILE *fp, double *a, int n, string s, bool col_mode){
	fprintf_real_array(fp, a, n, s, "%lg", col_mode);
}
void fprintf_real_array(string fname, double *a, int n, string s, string fmt_val, bool col_mode){
	FILE *fp = fopen(fname.c_str(), "w");
	fprintf_real_array(fp, a, n, s, fmt_val, col_mode);
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
	string fmt = fmt_val + "\n";
	for (size_t i = 0; i < m.size(); i++)
		fprintf(fp, fmt.c_str(), m[i][0], m[i][1], m[i][2]);
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
void printf_complex_mat(complex *a, int m, int n, string s){
	if (m*n <= 6){
		printf("%s", s.c_str());
		for (int i = 0; i < m; i++){
			for (int j = 0; j < n; j++)
				printf(" (%lg,%lg)", a[i*n + j].real(), a[i*n + j].imag());
			printf("; ");
		}
		printf("\n");
	}
	else{
		printf("%s\n", s.c_str());
		for (int i = 0; i < m; i++){
			for (int j = 0; j < n; j++)
				printf(" (%lg,%lg)", a[i*n + j].real(), a[i*n + j].imag());
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
void fprintf_complex_mat(FILE *fp, complex *a, int m, int n, string s){
	if (m * n <= 6){
		fprintf(fp, "%s", s.c_str());
		for (int i = 0; i < m; i++){
			for (int j = 0; j < n; j++)
				fprintf(fp, " (%lg,%lg)", a[i*n + j].real(), a[i*n + j].imag());
			fprintf(fp, "; ");
		}
		fprintf(fp, "\n");
	}
	else{
		fprintf(fp, "%s\n", s.c_str());
		for (int i = 0; i < m; i++){
			for (int j = 0; j < n; j++)
				fprintf(fp, " (%lg,%lg)", a[i*n + j].real(), a[i*n + j].imag());
			fprintf(fp, "\n");
		}
	}
}
void error_message(string s, string routine){
	printf((s+" in "+routine+"\n").c_str());
	exit(EXIT_FAILURE);
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

bool exists(string name){
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}

int last_file_index(string pre, string suf){
	int result = 0;
	while (true){
		ostringstream oss; oss << round(result + 1);
		string name = pre + oss.str() + suf;
		if (!exists(name)) return result;
		result++;
	}
}

bool is_dir(string name){
	struct stat buffer;
	if (stat(name.c_str(), &buffer) != 0)
		return false;
	else if (buffer.st_mode & S_IFDIR)
		return true;
	else
		return false;
}

string int2str(int i){
	ostringstream oss; oss << i;
	return oss.str();
}