/*-------------------------------------------------------------------
Copyright 2020 Ravishankar Sundararaman

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

#ifndef FEYNWANN_BLOCKCYCLICMATRIX_H
#define FEYNWANN_BLOCKCYCLICMATRIX_H
#ifdef SCALAPACK_ENABLED

#include <core/Util.h>
#include <algorithm>

//Wrapper to and extension of ScaLAPACK diagonalization routines
class BlockCyclicMatrix
{
public:
	typedef std::vector<double> Buffer; //!< used for local matrices and temporary working buffers for ScaLAPACK
	
	int N; //!< matrix dimension
	int blockSize; //!< block size for block-cyclic distribution
	MPIUtil* mpiUtil; //!< parent MPI communictor
	MPIUtil *mpiRow, *mpiCol; //!< cooommunictars within each process row and column
	int blacsContext;
	int desc[9]; //!< BLACS description of matrix distribution
	int nProcsRow, nProcsCol; //!< BLACS process grid dimensions
	int iProcRow, iProcCol; //!< Current process index in BLACS process grid
	int nRowsMine, nColsMine; //!< Number of rows and columns on current process
	size_t nDataMine; //!< Total number of local matrix entries
	std::vector<int> iRowsMine, iColsMine; //!< Indices of rows and columns that belong to current process
	std::vector<int> nRowsProc; //!< number of rows on each process
	
	BlockCyclicMatrix(int N, int blockSize, MPIUtil* mpiUtil); //!< Set up for diagnalization of NxN matrices parallelized over mpiUtil
	~BlockCyclicMatrix();
	
	//---- Diagonalization and helper routines ----
	
	//! Switch between multiple options for diagonalizing
	enum DiagMethod
	{	UsePDGEEVX, //!< Call pdgeevx available in recent versions of MKL ScaLAPACK only (broken in MKL so far)
		UsePDHSEQR, //!< Call pdhseqr along with Hessenberg and custom eigenvector extraction calls
		UsePDHSEQRm //!< Use a modified version of pdhseqr that reports intermediate progress
	};
	static EnumStringMap<DiagMethod> diagMethodMap;

	//! Diagonalize non-symmetric matrix A, returning eigenvalues and setting right and left eigenvectors in VR and VL.
	//! Balance the matrix for numerical stability if shouldBalance=true.
	//! The eigenvectors VR and VL are distributed the same way as A, and contain
	//! a single column for real eigenvalues and two consecutive columns of real
	//! and imaginary parts for complex eigenvalue pairs.
	//! The eigenvectors are normalized so that inv(VR) = dagger(VL), when interpreted as a complex matrix.
	//! (Correspondingly, VL^T * VR is a scalar matrix with 1 on the diagonals for real eigenvectors
	//! and 1/2 on the diagonals for the real and imaginary part columns of complex eigenvectors.)
	std::vector<complex> diagonalize(const Buffer& A, Buffer& VR, Buffer& VL, DiagMethod diagMethod, bool shouldBalance=true) const;
	
	//! Calculate and report errors in the computed eigenvalue decomposition
	void checkDiagonalization(const Buffer& A, const Buffer& VR, const Buffer& VL, const std::vector<complex>& E) const;
	
	//! Balance a matrix A by row/column scaling and return scale factors
	Buffer balance(Buffer& A) const;
	
	//! Hessenberg reduction of a matrix H in place, and return rotations Q (Hin = Q Hout Q^T)
	Buffer hessenberg(Buffer& H) const;
	
	//! Schur decomposition of a Hessenberg matrix H in place, and return eigenvalues
	//! At exit, H is replaced with a quasi-upper triangular matrix T
	//! and the rotations Q are updated such that A = Q T Q^T,
	//! where A = Q H Q^T is the original matrix that was converted to Hessenberg form.
	//! Use modified or original pdhseqr depending on diagMethod
	std::vector<complex> schur(Buffer& H, Buffer& Q, DiagMethod diagMethod) const;
	
	//! Compute right and left eigenvectors given Shur decomposition of a non-symmetric matrix (equivalent to LAPACK dtrevc)
	//! Input: upper quasi-triangular matrix T and orthogonal matrix Q, such that matrix A = Q T Q^T
	//! Output: right and left eigenvectors of A in VR and VL
	//! Optionally correct the eigenvectors for scale factors used to balance A is scaleFactors is non-null
	//! If evalSort is provided, sort eigenvectors to match the sorting of eigenvalues
	void getEvecs(const Buffer& T, const Buffer& Q, Buffer& VR, Buffer& VL, const Buffer* scaleFactors=0) const;
	
	//! C = beta C + alpha op(A) * op(B), where op = identity or transpose depending on transA and transB
	void matMult(double alpha, const Buffer& A, bool transA, const Buffer& B, bool transB, double beta, Buffer& C) const;

	//! C = A^T * B, where A is N x N, B is nRowsMine x nVec and C is nColsMine x nVec.
	//! Here, nVec is a certain number of vectors, typically << N.
	//! Note that B should contain all nVec vetcors irrespective of nColsMine,
	//! and the result in C contains all nVec vectors irrespective of nRowsMine
	void matMultVec(double alpha, const Buffer& A, const Buffer& B, Buffer& C) const;

	//---- I/O and debugging ----
	double matrixErr(const Buffer& A, const Buffer& B) const; //!< Calculate error between two distributed matrices
	double identityErr(const Buffer& A, double* offDiag=0) const; //!< Calculate error between a distributed matrix and identity (offDiag contains error in off-diagonal parts)
	void printMatrix(const Buffer& mat, const char* name="") const; //!< Synchronized print of all pieces of a distributed matrix
	void writeMatrix(const Buffer& mat, const char* fname) const; //!< Binary-write matrix to file
	void testRandom(DiagMethod diagMethod, double fillFactor) const; //!< Test specified diagonalization method with a random matrix with specified fill factor
	
	//---- Indexing utilties ----
	
	//! Get index into local storage given global indices and dimensions
	//! Returns -1 if corresponding value does not belong to current process
	inline int localIndex(int iRow, int iCol) const;
	
	//! Return process number where this entry will be local, and set corresponding localIndex in localIndex
	inline int globalIndex(int iRow, int iCol, int& localIndex) const;
	
	//! Get the range of indices [iMineStart,iMineStop) to sorted array iMine that have values in range [iStart,iStop)
	inline void getRange(const std::vector<int>& iMine, int iStart, int iStop, int& iMineStart, int& iMineStop) const;
	
	//! Return index of i in sorted array iMine, and -1 if not found
	inline int localIndex1D(const std::vector<int>& iMine, int i) const;
	inline int localRowIndex(int iRow) const { return localIndex1D(iRowsMine, iRow); } //!< Return local index of row, and -1 if not found
	inline int localColIndex(int iCol) const { return localIndex1D(iColsMine, iCol); } //!< Return local index of column, and -1 if not found
};


//------ Inline function implementations ------

inline int BlockCyclicMatrix::localIndex(int iRow, int iCol) const
{	//Identify row and column indices:
	#define InitIndices(dim) \
		int iBlock##dim##Global = i##dim / blockSize; \
		if(iBlock##dim##Global % nProcs##dim != iProc##dim) return -1; \
		int iBlock##dim = iBlock##dim##Global / nProcs##dim; /*local block index*/ \
		int iElem##dim = i##dim % blockSize; /*index within block*/ \
		int i##dim##Mine = iBlock##dim * blockSize + iElem##dim;
	InitIndices(Row)
	InitIndices(Col)
	#undef InitIndices
	//Compute flattened local index:
	return iColMine*nRowsMine + iRowMine;
}

inline int BlockCyclicMatrix::globalIndex(int iRow, int iCol, int& localIndex) const
{	//Split row and column indices, each into process index and local index:
	#define InitIndices(dim) \
		int iBlock##dim##Global = i##dim / blockSize; \
		int whose##dim = iBlock##dim##Global % nProcs##dim; /*process index*/ \
		int iBlock##dim = iBlock##dim##Global / nProcs##dim; /*local block index*/ \
		int iElem##dim = i##dim % blockSize; /*index within block*/ \
		int i##dim##Local = iBlock##dim * blockSize + iElem##dim;
	InitIndices(Row)
	InitIndices(Col)
	#undef InitIndices
	//Compute flattened processor index and corresponding local index:
	int whose = whoseRow*nProcsCol + whoseCol; //row-major process mapping
	localIndex = iRowLocal + iColLocal*nRowsProc[whose]; //column-major data mapping in each process
	return whose;
}

inline void BlockCyclicMatrix::getRange(const std::vector<int>& iMine, int iStart, int iStop, int& iMineStart, int& iMineStop) const
{	iMineStart = std::lower_bound(iMine.begin(), iMine.end(), iStart) - iMine.begin();
	iMineStop = std::lower_bound(iMine.begin(), iMine.end(), iStop) - iMine.begin();
}

inline int BlockCyclicMatrix::localIndex1D(const std::vector<int>& iMine, int i) const
{	auto iter = std::lower_bound(iMine.begin(), iMine.end(), i);
	return ((iter!=iMine.end()) and (*iter == i)) ? (iter-iMine.begin()) : -1;
}

#endif //SCALAPACK_ENABLED
#endif // FEYNWANN_BLOCKCYCLICMATRIX_H
