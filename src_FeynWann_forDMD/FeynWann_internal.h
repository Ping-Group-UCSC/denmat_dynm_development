/*-------------------------------------------------------------------
Copyright 2022 Ravishankar Sundararaman

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

#ifndef FEYNWANN_FEYNWANN_INTERNAL_H
#define FEYNWANN_FEYNWANN_INTERNAL_H

#include <core/matrix.h>


//Get iMatrix'th matrix of specified dimensions from pointer src, assuming they are stored contiguously there in column-major order)
inline matrix getMatrix(const complex* src, int nRows, int nCols, int iMatrix=0)
{	matrix result(nRows, nCols);
	eblas_copy(result.data(), src + iMatrix*result.nData(), result.nData());
	return result;
}


//Prepare and broadcast matrices on custom communicator:
inline void bcast(diagMatrix& m, int nRows, MPIUtil* mpiUtil, int root)
{	m.resize(nRows);
	mpiUtil->bcast(m.data(), nRows, root);
}
inline void bcast(matrix& m, int nRows, int nCols, MPIUtil* mpiUtil, int root)
{	if(!m) m.init(nRows, nCols);
	mpiUtil->bcast(m.data(), m.nData(), root);
}


template<typename T> vector3<T> elemwiseProd(vector3<int> a, vector3<T> b)
{	return vector3<T>(a[0]*b[0], a[1]*b[1], a[2]*b[2]);
}


//Loop i till iStop, sampling a 3D mesh of dimensions S
//At each point, set fractional coordinates x offset by x0 and run code
#define PartialLoop3D(S, i, iStop, x, x0, code) \
	vector3<int> i##v( \
		i / (S[2]*S[1]), \
		(i/S[2]) % S[1], \
		i % S[2] ); \
	vector3<> i##Frac(1./S[0], 1./S[1], 1./S[2]); \
	while(i<iStop) \
	{	\
		x = x0 + vector3<>(i##v[0]*i##Frac[0], i##v[1]*i##Frac[1], i##v[2]*i##Frac[2]); \
		code \
		\
		i++; if(i==iStop) break; \
		i##v[2]++; \
		if(i##v[2]==S[2]) \
		{	i##v[2]=0; \
			i##v[1]++; \
			if(i##v[1]==S[1]) \
			{	i##v[1] = 0; \
				i##v[0]++; \
			} \
		} \
	}



//! Read cell map
std::vector<vector3<int>> readCellMap(string fname);


//! Read cell weights
std::vector<matrix> readCellWeights(string fname, int nCells, int nAtoms, int nBands);


//! Read phonon basis file
diagMatrix readPhononBasis(string fname);
std::vector<vector3<>> readPhononBasisFromFile(string fname);//JX


//! Read an array of vector3<> from a plain text file (implemented in wannier/WannierMinimizer_phonon.cpp)
std::vector<vector3<>> readArrayVec3(string fname); 


#endif //FEYNWANN_FEYNWANN_INTERNAL_H
