/*-------------------------------------------------------------------
Copyright 2019 Ravishankar Sundararaman

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

#ifndef FEYNWANN_SPARSEMATRIX_H
#define FEYNWANN_SPARSEMATRIX_H

#include <core/matrix.h>

//----- Rudimentary triplet format square sparse matrix with restricted but fast, inline operations ----
struct SparseEntry
{	int i, j;
	complex val;
};
typedef std::vector<SparseEntry> SparseMatrix;

//Multiply dagger(S)*M*S for sparse matrix S and dense matrix M
inline SparseMatrix SdagMS(const SparseMatrix& S, const matrix& M)
{	SparseMatrix result; result.reserve(S.size()*S.size());
	const complex* m = M.data();
	for(const SparseEntry& s1: S)
		for(const SparseEntry& s2: S)
		{	SparseEntry sr;
			sr.i = s1.j;
			sr.j = s2.j;
			sr.val = s1.val.conj() * m[M.index(s1.i,s2.i)] * s2.val;
			result.push_back(sr);
		}
	return result;
}

//Multiply S*M*dagger(S) for sparse matrix S and dense matrix M
inline SparseMatrix SMSdag(const SparseMatrix& S, const matrix& M)
{	SparseMatrix result; result.reserve(S.size()*S.size());
	const complex* m = M.data();
	for(const SparseEntry& s1: S)
		for(const SparseEntry& s2: S)
		{	SparseEntry sr;
			sr.i = s1.i;
			sr.j = s2.i;
			sr.val = s1.val * m[M.index(s1.j,s2.j)] * s2.val.conj();
			result.push_back(sr);
		}
	return result;
}

//Extract diagonal part of product of sparse matrices:
inline diagMatrix diagSS(const SparseMatrix& S1, const SparseMatrix& S2, int N)
{	diagMatrix result(N);
	for(const SparseEntry& s1: S1)
		for(const SparseEntry& s2: S2)
			if(s1.i==s2.j && s1.j==s2.i)
				result[s1.i] += (s1.val * s2.val).real();
	return result;
}

//Multiply sparse matrix with dense matrix on left:
inline matrix operator*(const matrix& M, const SparseMatrix& S)
{	int N = M.nRows(); //assumed square
	matrix R = zeroes(N, N);
	complex* r = R.data();
	const complex* m = M.data();
	for(const SparseEntry& s: S)
	{	complex* rCur = r + N*s.j;
		const complex* mCur = m + N*s.i;
		for(int k=0; k<N; k++)
			*(rCur++) += *(mCur++) * s.val;
	}
	return R;
}

//Multiply sparse matrix with dense matrix on right:
inline matrix operator*(const SparseMatrix& S, const matrix& M)
{	int N = M.nRows(); //assumed square
	matrix R = zeroes(N, N);
	complex* r = R.data();
	const complex* m = M.data();
	for(const SparseEntry& s: S)
	{	complex* rCur = r + s.i;
		const complex* mCur = m + s.j;
		int offset = 0;
		for(int k=0; k<N; k++)
		{	rCur[offset] += s.val * mCur[offset];
			offset += N;
		}
	}
	return R;
}

#endif //FEYNWANN_SPARSEMATRIX_H
