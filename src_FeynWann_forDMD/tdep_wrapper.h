/*-------------------------------------------------------------------
Copyright 2022 Josh Quinton, Ravishankar Sundararaman

This file is part of JDFTx.

JDFTx is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

JDFTx is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with JDFTx.  If not, see <http://www.gnu.org/licenses/>.
-------------------------------------------------------------------*/

#ifndef FEYNWANN_TDEP_WRAPPER_H_
#define FEYNWANN_TDEP_WRAPPER_H_

#include <core/scalar.h>

extern "C"
{
	int tdep_initialize_(int* verbosity);
	int tdep_compute_(double* qcart, double* omega, complex* U);
}

#endif //FEYNWANN_TDEP_WRAPPER_H_
