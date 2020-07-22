#pragma once
#include <core/scalar.h>
#include "FeynWann.h"

const complex c0(0, 0);
const complex c1(1, 0);
const complex cm1(-1, 0);
const complex ci(0, 1);
const complex cmi(0, -1);

double compute_sz(complex **dm, size_t nk, size_t nkTot, int nb, int bStart, int bStop, std::vector<FeynWann::StateE>& e);
vector3<> compute_spin(std::vector<std::vector<matrix>> m, size_t nk, size_t nkTot, int nb, int bStart, int bStop, std::vector<FeynWann::StateE>& e);
void init_dm(complex **dm, size_t nk, int nb, std::vector<diagMatrix>& F);
void set_dm1(complex **dm, size_t nk, int nb, complex **dm1);

double** alloc_real_array(int n1, int n2, double val = 0.);
complex* alloc_array(int n1, complex val = c0);
complex** alloc_array(int n1, int n2, complex val = c0);
complex*** alloc_array(int n1, int n2, int n3, complex val = c0);

void zeros(double* arr, int n1);
void zeros(double** arr, int n1, int n2);
void zeros(complex* arr, int n1);
void zeros(complex** arr, int n1, int n2);
void zeros(complex*** arr, int n1, int n2, int n3);
void zeros(matrix& m, int n);
void zeros(std::vector<matrix>& v, int n);
void error_message(string s, string routine);