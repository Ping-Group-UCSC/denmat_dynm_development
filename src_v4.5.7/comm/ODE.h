#pragma once
#include <scalar.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>

static void copy_complex_from_real(complex **a, const double b[], size_t n){ // n is size of a
	for (int i = 0; i < n; i++)
		a[0][i] = complex(b[2 * i], b[2 * i + 1]); // memory of 2D array a must be continous
}
static void copy_real_from_complex(double b[], complex **a, size_t n){ // n is size of a
	for (int i = 0; i < n; i++){
		b[2 * i] = a[0][i].real(); // memory of 2D array a must continous
		b[2 * i + 1] = a[0][i].imag();
	}
}

struct ODEparameters{
public:
	int ncalls;
	double hstart, hmin, hmax, hmax_laser, epsabs;
};

extern ODEparameters ode;