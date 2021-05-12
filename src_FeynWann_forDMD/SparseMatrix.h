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
	//Accessors supporting dagger operation:
	template<bool dagger=false> int I() const { return dagger ? j : i; }
	template<bool dagger=false> int J() const { return dagger ? i : j; }
	template<bool dagger=false> complex Val() const { return dagger ? val.conj() : val; }
};
struct SparseMatrix : public std::vector<SparseEntry>
{
private:
	int nr, nc;
public:
    inline SparseMatrix(int nRows=0, int nCols=0, int nNZexpected=0) { init(nRows, nCols, nNZexpected); }
	inline void init(int nRows, int nCols, int nNZexpected=0)
	{	nr = nRows; nc = nCols;
		if(nNZexpected) reserve(nNZexpected);
	}
	//Accessors supporting dagger operation:
	template<bool dagger=false> int nRows() const { return dagger ? nc : nr; }
	template<bool dagger=false> int nCols() const { return dagger ? nr : nc; }
};

//Accumulate S1 * M1 * S2 * M2 for sparse matrices S1 and S2 (each optionally daggered), and dense matrices M1 and M2
template<bool dag1, bool dag2> void axpySMSM(double alpha, const SparseMatrix& S1, const matrix& M1, const SparseMatrix& S2, const matrix& M2, matrix& R)
{	assert(R.nRows() == S1.nRows<dag1>());
	assert(S1.nCols<dag1>() == M1.nRows());
	assert(M1.nCols() == S2.nRows<dag2>());
	assert(S2.nCols<dag2>() == M2.nRows());
	assert(M2.nCols() == R.nCols());
	const complex* m1 = M1.data();
	const complex* m2 = M2.data();
	complex* r = R.data();
	for(const SparseEntry& s1: S1)
		for(const SparseEntry& s2: S2)
		{	//Get sparse entry corresponding to alpha * S1 * M1 * S2:
			int i = s1.I<dag1>();
			int j = s2.J<dag2>();
			complex val = alpha * s1.Val<dag1>() * m1[M1.index(s1.J<dag1>(),s2.I<dag2>())] * s2.Val<dag2>();
			//Implement its multiplication with M2 on the right:
			complex* rCur = r + i;
			const complex* m2cur = m2 + j;
			for(int k=0; k<R.nCols(); k++)
			{	(*rCur) += val * (*m2cur);
				rCur += R.nRows();
				m2cur += M2.nRows();
			}
		}
}

//Accumulate M1 * S1 * M2 * S2 for sparse matrices S1 and S2 (each optionally daggered), and dense matrices M1 and M2
template<bool dag1, bool dag2> void axpyMSMS(double alpha, const matrix& M1, const SparseMatrix& S1, const matrix& M2, const SparseMatrix& S2, matrix& R)
{	int nRows = R.nRows();
	assert(R.nRows() == M1.nRows());
	assert(M1.nCols() == S1.nRows<dag1>());
	assert(S1.nCols<dag1>() == M2.nRows());
	assert(M2.nCols() == S2.nRows<dag2>());
	assert(S2.nCols<dag2>() == R.nCols());
	const complex* m1 = M1.data();
	const complex* m2 = M2.data();
	complex* r = R.data();
	for(const SparseEntry& s1: S1)
		for(const SparseEntry& s2: S2)
		{	//Get sparse entry corresponding to alpha * S1 * M2 * S2:
			int i = s1.I<dag1>();
			int j = s2.J<dag2>();
			complex val = alpha * s1.Val<dag1>() * m2[M2.index(s1.J<dag1>(),s2.I<dag2>())] * s2.Val<dag2>();
			//Implement its multiplication with M1 on the left:
			complex* rCur = r + nRows*j;
			const complex* m1cur = m1 + nRows*i;
			for(int k=0; k<nRows; k++)
				*(rCur++) += *(m1cur++) * val;
		}
}


//Extract diagonal part of product of sparse matrices:
inline diagMatrix diagSS(const SparseMatrix& S1, const SparseMatrix& S2)
{	assert(S1.nCols() == S2.nRows()); //for S1 * S2 to be meaningful
	assert(S1.nRows() == S2.nCols()); //for result to be square
	diagMatrix result(S1.nRows());
	for(const SparseEntry& s1: S1)
		for(const SparseEntry& s2: S2)
			if(s1.i==s2.j && s1.j==s2.i)
				result[s1.i] += (s1.val * s2.val).real();
	return result;
}

//Multiply sparse matrix with dense matrix on left:
inline matrix operator*(const matrix& M, const SparseMatrix& S)
{	assert(M.nCols() == S.nRows());
	int nRows = M.nRows();
	matrix R = zeroes(M.nRows(), S.nCols());
	complex* r = R.data();
	const complex* m = M.data();
	for(const SparseEntry& s: S)
	{	complex* rCur = r + nRows*s.j;
		const complex* mCur = m + nRows*s.i;
		for(int k=0; k<nRows; k++)
			*(rCur++) += *(mCur++) * s.val;
	}
	return R;
}

//Multiply sparse matrix with dense matrix on right:
inline matrix operator*(const SparseMatrix& S, const matrix& M)
{	assert(S.nCols() == M.nRows());
	int nCols = M.nCols();
	matrix R = zeroes(S.nRows(), M.nCols());
	complex* r = R.data();
	const complex* m = M.data();
	for(const SparseEntry& s: S)
	{	complex* rCur = r + s.i;
		const complex* mCur = m + s.j;
		for(int k=0; k<nCols; k++)
		{	(*rCur) += s.val * (*mCur);
			rCur += R.nRows();
			mCur += M.nRows();
		}
	}
	return R;
}

#endif //FEYNWANN_SPARSEMATRIX_H
