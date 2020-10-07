#include <myio.h>

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
	printf((s+" in "+routine).c_str());
	exit(EXIT_FAILURE);
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

void fseek_bigfile(FILE *fp, size_t count, size_t size){
	size_t count_step = 100000000 / size;
	size_t nsteps = count / count_step;
	size_t reminder = count % count_step;
	size_t pos = reminder * size;
	size_t accum = reminder;
	fseek(fp, pos, SEEK_SET);
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