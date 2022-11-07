#pragma once
#include <sstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <string>
#include <scalar.h>
#include <vector3.h>
using namespace std;

void printf_real_array(double *a, int n, string s, bool col_mode=false);
void fprintf_real_array(FILE *fp, double *a, int n, string s, bool col_mode=false);
void fprintf_real_array(string fname, double *a, int n, string s, bool col_mode = false);
void fprintf_real_array(FILE *fp, double *a, int n, string s, string fmt_val, bool col_mode = false);
void fprintf_real_array(string fname, double *a, int n, string s, string fmt_val, bool col_mode = false);
void fprintf(FILE *fp, std::vector<vector3<>> m, string s, string fmt_val = " %lg %lg %lg");
void fprintf(string fname, std::vector<vector3<>> m, string s, string fmt_val = " %lg %lg %lg");

void printf_complex_mat(complex *m, int n, string s);

void fprintf_complex_mat(FILE *fp, complex *m, int n, string s);

void printf_complex_mat(complex *a, int m, int n, string s);

void fprintf_complex_mat(FILE *fp, complex *a, int m, int n, string s);

void error_message(string s, string routine="");

size_t file_size(FILE *fp);
void check_file_size(FILE *fp, size_t expect_size, string message);
bool check_file_size(FILE *fp, size_t expect_size);

void fseek_bigfile(FILE *fp, size_t count, size_t size, int origin=SEEK_SET);

bool exists(string name);

int last_file_index(string pre, string suf);

bool is_dir(string name);

string int2str(int i);