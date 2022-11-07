#ifdef SCALAPACK_ENABLED
#include "BlockCyclicMatrix.h"
#include <core/Util.h>
#include <core/BlasExtra.h>
#include <mkl_blacs.h>
#include <mkl_pblas.h>
#include <mkl_scalapack.h>
#include <mkl_lapack.h>

EnumStringMap<BlockCyclicMatrix::DiagMethod> BlockCyclicMatrix::diagMethodMap(
	BlockCyclicMatrix::UsePDGEEVX, "PDGEEVX",
	BlockCyclicMatrix::UsePDHSEQR, "PDHSEQR",
	BlockCyclicMatrix::UsePDHSEQRm, "PDHSEQRm"
);

//Return list of indices in a given dimension (row or column) that belong to me in block-cyclic distribution
std::vector<int> distributedIndices(int nTotal, int blockSize, int iProcDim, int nProcsDim)
{	int zero = 0;
	int nMine = numroc_(&nTotal, &blockSize, &iProcDim, &zero, &nProcsDim);
	std::vector<int> myIndices; myIndices.reserve(nMine);
	int blockStride = blockSize * nProcsDim;
	int nBlocksMineMax = (nTotal + blockStride - 1) / blockStride;
	for(int iBlock=0; iBlock<nBlocksMineMax; iBlock++)
	{	int iStart = iProcDim*blockSize + iBlock*blockStride;
		int iStop = std::min(iStart+blockSize, nTotal);
		for(int i=iStart; i<iStop; i++)
			myIndices.push_back(i);
	}
	assert(int(myIndices.size()) == nMine);
	return myIndices;
}

BlockCyclicMatrix::BlockCyclicMatrix(int N, int blockSize, MPIUtil* mpiUtil) : N(N), blockSize(blockSize), mpiUtil(mpiUtil)
{
	//Calculate squarest possible process grid:
	int nProcesses = mpiUtil->nProcesses();
	nProcsRow = int(round(sqrt(nProcesses)));
	while(nProcesses % nProcsRow) nProcsRow--;
	nProcsCol = nProcesses / nProcsRow;
	//Initialize BLACS process grid:
	{	int unused=-1, what=0;
		blacs_get_(&unused, &what, &blacsContext);
		blacs_gridinit_(&blacsContext, "Row-major", &nProcsRow, &nProcsCol);
		blacs_gridinfo_(&blacsContext, &nProcsRow, &nProcsCol, &iProcRow, &iProcCol);
		assert(mpiUtil->iProcess() == iProcRow * nProcsCol + iProcCol); //this mapping is assumed below, so check
	}
	//Initialize row and column communicators:
	mpiRow = new MPIUtil(0, NULL, MPIUtil::ProcDivision(mpiUtil, 0, iProcRow)); assert(mpiRow->iProcess() == iProcCol);
	mpiCol = new MPIUtil(0, NULL, MPIUtil::ProcDivision(mpiUtil, 0, iProcCol)); assert(mpiCol->iProcess() == iProcRow);
	logPrintf("Initialized %d x %d process BLACS grid.\n", nProcsRow, nProcsCol);
	
	//Initialize matrix distribution:
	logPrintf("Setting up ScaLAPACK matrix with dimension %d\n", N); logFlush();
	if(N <= blockSize * (std::max(nProcsRow, nProcsCol) - 1))
		die("No data on some processes: reduce blockSize or # processes.\n");
	iRowsMine = distributedIndices(N, blockSize, iProcRow, nProcsRow); //indices of rows on current process
	iColsMine = distributedIndices(N, blockSize, iProcCol, nProcsCol); //indices of cols on current process
	nRowsMine = iRowsMine.size();
	nColsMine = iColsMine.size();
	nDataMine = nRowsMine * nColsMine;
	{	int zero=0, info;
		descinit_(desc, &N, &N, &blockSize, &blockSize, &zero, &zero, &blacsContext, &nRowsMine, &info);
		assert(info==0);
	}
	
	//Make number of rows on each process globally available:
	nRowsProc.assign(mpiUtil->nProcesses(), 0);
	nRowsProc[mpiUtil->iProcess()] = nRowsMine;
	mpiUtil->allReduceData(nRowsProc, MPIUtil::ReduceMax);
}

BlockCyclicMatrix::~BlockCyclicMatrix()
{	delete mpiRow;
	delete mpiCol;
}

//Calculate error between distributed matrices
double BlockCyclicMatrix::matrixErr(const Buffer& A, const Buffer& B) const
{	double errSq = 0.;
	for(size_t i=0; i<nDataMine; i++)
		errSq += std::pow(A[i]-B[i], 2);
	mpiUtil->allReduce(errSq, MPIUtil::ReduceSum);
	return sqrt(errSq/(N*N));
}

//Calculate error between a distributed matrix and identity
double BlockCyclicMatrix::identityErr(const Buffer& A, double* offDiagErr) const
{	double errSq = 0., offErrSq = 0.;
	const double* Aptr = A.data();
	for(int iCol: iColsMine)
		for(int iRow: iRowsMine)
		{	double errSqCur = std::pow(*(Aptr++) - (iRow==iCol ? 1. : 0.), 2);
			errSq += errSqCur;
			if(iRow!=iCol) offErrSq += errSqCur;
		}
	mpiUtil->allReduce(errSq, MPIUtil::ReduceSum);
	mpiUtil->allReduce(offErrSq, MPIUtil::ReduceSum);
	if(offDiagErr) *offDiagErr = sqrt(offErrSq/(N*N));
	return sqrt(errSq/(N*N));
}

//Print all pieces of distributed block cyclic matrix
void BlockCyclicMatrix::printMatrix(const Buffer& mat, const char* name) const
{	ostringstream oss; 
	oss << "\nOn process (" << iProcRow << ',' << iProcCol << ":\n";
	for(int iRow=0; iRow<N; iRow++)
	{	for(int iCol=0; iCol<N; iCol++)
		{	int index = localIndex(iRow, iCol);
			oss.width(9);
			oss.precision(5);
			if(index<0) oss << "########"; else oss << std::fixed << mat[index];
		}
		oss << '\n';
	}
	string buf = oss.str();
	if(mpiUtil->isHead())
	{	if(strlen(name)>0)
			logPrintf("\n----------- Matrix %s -----------\n", name);
		for(int iProc=0; iProc<mpiUtil->nProcesses(); iProc++)
		{	if(iProc) mpiUtil->recv(buf, iProc, 0);
			logPrintf("%s", buf.c_str());
		}
		logFlush();
	}
	else mpiUtil->send(buf, 0, 0);
}

void BlockCyclicMatrix::writeMatrix(const Buffer& mat, const char* fname) const
{	assert(mat.size() == nDataMine);
	int nBlocks = ceildiv(N, blockSize);
	int blockInterval = std::max(1, int(round(nBlocks/50.))); //interval for reporting progress
	FILE* fp = 0;
	BlockCyclicMatrix::Buffer buf, bufCur(blockSize*blockSize);
	if(mpiUtil->isHead())
	{	fp = fopen(fname, "w");
		buf.resize(N * blockSize); //all rows of a block of columns
	}
	//Outer loop over column blocks:
	for(int jBlock=0; jBlock<nBlocks; jBlock++)
	{	int jWhose = jBlock % nProcsCol;
		int jStart = jBlock*blockSize;
		int jStop = std::min((jBlock+1)*blockSize, N);
		int jCount = jStop - jStart;
		int jLocal = (jBlock / nProcsCol) * blockSize;
		//Inner loop over row blocks:
		for(int iBlock=0; iBlock<nBlocks; iBlock++)
		{	int iWhose = iBlock % nProcsRow;
			int iStart = iBlock*blockSize;
			int iStop = std::min((iBlock+1)*blockSize, N);
			int iCount = iStop - iStart;
			int iLocal = (iBlock / nProcsRow) * blockSize;
			int whose = jWhose + iWhose * nProcsCol;
			int count = jCount*iCount;
			if(mpiUtil->isHead())
			{	if(whose)
				{	//Recv data from process that owns this block and copy to output buffer:
					mpiUtil->recv(bufCur.data(), count, whose, iBlock+jBlock*nBlocks);
					dlacpy_("A", &iCount, &jCount, bufCur.data(),&iCount, &buf[iStart],&N);
				}
				else //Direct local copy of block to output buffer:
					dlacpy_("A", &iCount, &jCount, &mat[iLocal+jLocal*nRowsMine],&nRowsMine, &buf[iStart],&N);
			}
			else if(whose == mpiUtil->iProcess())
			{	//Send data to head:
				dlacpy_("A", &iCount, &jCount, &mat[iLocal+jLocal*nRowsMine],&nRowsMine, bufCur.data(),&iCount);
				mpiUtil->send(bufCur.data(), count, 0, iBlock+jBlock*nBlocks);
			}
		}
		if(mpiUtil->isHead()) fwrite(buf.data(), sizeof(double), N*jCount, fp);
		if((jBlock+1)%blockInterval==0) { logPrintf("%d%% ", int(round((jBlock+1)*100./nBlocks))); logFlush(); }
	}
	if(mpiUtil->isHead()) fclose(fp);
}

//Test with a random matrix:
void BlockCyclicMatrix::testRandom(DiagMethod diagMethod, double fillFactor) const
{
	//Create and diagonalize test matrix:
	Buffer A(nDataMine), VR, VL; 
	double* Adata = A.data();
	for(int iCol: iColsMine)
		for(int iRow: iRowsMine)
		{	//Simple reproducible xorshift RNG:
			uint32_t x = iRow + N*iCol;
			x ^= x << 13; x ^= x >> 17; x ^= x << 5; double f = 2.3283e-10*x; //in [0,1)
			x ^= x << 13; x ^= x >> 17; x ^= x << 5; double val = 2.3283e-10*x - 0.5; //in [-0.5,0.5)
			*(Adata++) = (f < fillFactor) ? val : 0.;
		}
	Buffer Acopy(A); //run diagonalization on a destructible copy
	std::vector<complex> E = diagonalize(Acopy, VR, VL, diagMethod);
	
	//Determine error in eigen-decomposition:
	checkDiagonalization(A, VR, VL, E);
}

void BlockCyclicMatrix::checkDiagonalization(const Buffer& A, const Buffer& VR, const Buffer& VL, const std::vector<complex>& E) const
{
	//Get matrix norm:
	int one = 1;
	double Anorm = pdlange_("F", &N, &N, A.data(), &one, &one, desc, NULL);
	
	//Report eigenvalue statistics:
	int nReal = 0;
	double EreMin = +DBL_MAX, EreMax = -DBL_MAX, EreMean = 0., EreSqSum = 0.;
	double EimMin = +DBL_MAX, EimMax = -DBL_MAX, EimMean = 0., EimSqSum = 0.;
	for(const complex& e: E)
	{	double re = e.real(), im = fabs(e.imag());
		if(im < 1e-14*Anorm) nReal++;
		EreMin = std::min(EreMin, re); EreMax = std::max(EreMax, re);
		EimMin = std::min(EimMin, im); EimMax = std::max(EimMax, im);
		EreMean += re; EreSqSum += std::pow(re,2);
		EimMean += im; EimSqSum += std::pow(im,2);
	}
	EreMean /= N; EimMean /= N;
	double EreStd = sqrt(EreSqSum/N - std::pow(EreMean,2));
	double EimStd = sqrt(EimSqSum/N - std::pow(EimMean,2));
	logPrintf("Eigenvalue statistics:\n");
	logPrintf("\t%d real and %d complex pairs\n", nReal, (N-nReal)/2);
	logPrintf("\tReal part:  min: %9.3lg  max: %9.3lg  mean: %9.3lg  std: %9.3lg\n", EreMin, EreMax, EreMean, EreStd);
	logPrintf("\t|Im part|:  min: %9.3lg  max: %9.3lg  mean: %9.3lg  std: %9.3lg\n", EimMin, EimMax, EimMean, EimStd);
	
	//Check VL-VR overlap:
	BlockCyclicMatrix::Buffer O;
	matMult(1., VL,true, VR,false, 0., O);
	logPrintf("RMSE VL^VR: %le\n", identityErr(O));
	//printMatrix(O, "O");
	
	//Form matrix of eigenvalues:
	BlockCyclicMatrix::Buffer Emat(nDataMine);
	double* Edata = Emat.data();
	for(int iCol: iColsMine)
		for(int iRow: iRowsMine)
		{	double val = 0.;
			if(iRow==iCol) val = E[iRow].real();
			if((iRow==iCol+1) and (E[iRow].imag()<0.)) val = E[iRow].imag();
			if((iRow==iCol-1) and (E[iRow].imag()>0.)) val = E[iRow].imag();
			*(Edata++) = val;
		}
	BlockCyclicMatrix::Buffer lhs(nDataMine), rhs(nDataMine);
	
	//Right eigenvector error report:
	matMult(1., A,false, VR,false, 0., lhs);
	matMult(1., VR,false, Emat,false, 0., rhs);
	logPrintf("RMS A*VR-VR*E: %le\n", matrixErr(lhs,rhs));
	
	//Left eigenvector error report:
	matMult(1., VL,true, A,false, 0., lhs);
	matMult(1., Emat,false, VL,true, 0., rhs);
	logPrintf("RMS VL'*A-E*VL': %le\n", matrixErr(lhs,rhs));
}


std::vector<complex> BlockCyclicMatrix::diagonalize(const Buffer& A, Buffer& VR, Buffer& VL, DiagMethod diagMethod, bool shouldBalance) const
{	static StopWatch watch("BlockCyclicMatrix::diagonalize"); watch.start();
	Buffer H(A); //modifiable copy that is balanced, Hessenberg'd and Schur'd
	std::vector<complex> evals; //resulting eigenvalues
	switch(diagMethod)
	{	case UsePDGEEVX:
		{	die("\nMKL pdgeevx is currently broken (version 2020.2) and always segfaults when computing eigenvectors.\n"
			"Try uncommenting the following code with the next MKL release, and complete implementation if it does.\n\n");
			/*
			VR.resize(nDataMine);
			VL.resize(nDataMine);
			int iLo = 1, iHi = N;
			Buffer scale(N, 1.), rconde(N), wr(N), wi(N); //scale factors, condition numbers, eigenvalue components
			double abnrm = 0.; //norm of matrix returned below
			Buffer work(1); int lwork = -1, info = 0;
			logPrintf("Diagonalizing matrix ... "); logFlush();
			for(int pass=0; pass<2; pass++) //first pass is workspace query, next pass is actual calculation
			{	pdgeevx_(shouldBalance ? "B" : "N", "V", "V", "N", &N,
					H.data(), desc, wr.data(), wi.data(), VL.data(), desc, VR.data(), desc,
					&iLo, &iHi, scale.data(), &abnrm,  rconde.data(), NULL,
					work.data(), &lwork, &info);
				if(info < 0)
				{	int errCode = -info;
					if(errCode < 100) die("Error in argument# %d to pdgeevx.\n", errCode)
					else die("Error in entry %d of argument# %d to pdgeevx.\n", errCode%100, errCode/100)
				}
				if(pass) break; //done
				//After first-pass, use results of work-space query to allocate:
				lwork = int(work.data()[0]);
				work.resize(lwork);
			}
			logPrintf("done at t[s]: %.2lf.\n", clock_sec());
			//Normalize eigenvectors:
			logPrintf("Normalizing eigenvectors ... "); logFlush();
			//TODO: implement eigenvector normalization if this call ever succeeds
			logPrintf("done at t[s]: %.2lf.\n", clock_sec());
			//Collect eigenvalues into complex array:
			evals.resize(N);
			for(int i=0; i<N; i++)
				evals[i] = complex(wr[i], wi[i]);
			*/
			break;
		}
		case UsePDHSEQR:
		case UsePDHSEQRm:
		{	Buffer scale; //optional scale factors
			if(shouldBalance) scale = balance(H); //Balance matrix
			Buffer Q = hessenberg(H); //Hessenberg reduction
			evals = schur(H, Q, diagMethod); //Schur decomposition and eigenvalues
			getEvecs(H, Q, VR, VL, shouldBalance ? &scale : NULL); //Transform Schur vectors to eigenvectors
			break;
		}
	}
	watch.stop();
	return evals;
}


BlockCyclicMatrix::Buffer BlockCyclicMatrix::balance(Buffer& A) const
{	static StopWatch watch("BlockCyclicMatrix::balance"); watch.start();
	assert(A.size()==nDataMine); 
	int iLo = 1, iHi = N;
	Buffer scaleFactors(N, 1.);
	logPrintf("Balancing matrix ... "); logFlush();
	int info = 0;
	pdgebal_("Scale", &N, A.data(), desc, &iLo, &iHi, scaleFactors.data(), &info);
	if(info < 0) die("Error in argument# %d to pdgebal.\n", -info);
	//Report range of scale factors:
	double scaleMin = +DBL_MAX, scaleMax = -DBL_MAX;
	for(const double s: scaleFactors)
	{	scaleMin = std::min(s, scaleMin);
		scaleMax = std::max(s, scaleMax);
	}
	logPrintf("done at t[s]: %.2lf. Scale factor range: [ %lg , %lg ]\n", clock_sec(), scaleMin, scaleMax);
	watch.stop();
	return scaleFactors;
}

BlockCyclicMatrix::Buffer BlockCyclicMatrix::hessenberg(Buffer& H) const
{	static StopWatch watch("BlockCyclicMatrix::hessenberg"); watch.start();
	assert(H.size()==nDataMine);
	
	//Hessenberg reduction by Householder transformations:
	int iLo = 1, iHi = N;
	int lwork = -1, one = 1, info = 0;
	Buffer work(1), tau(nColsMine);
	logPrintf("Hessenberg reduction ... "); logFlush();
	for(int pass=0; pass<2; pass++) //first pass is workspace query, next pass is actual calculation
	{	pdgehrd_(&N, &iLo, &iHi, H.data(), &one, &one, desc, tau.data(), work.data(), &lwork, &info);
		if(info < 0)
		{	int errCode = -info;
			if(errCode < 100) die("Error in argument# %d to pdgehrd.\n", errCode)
			else die("Error in entry %d of argument# %d to pdgehrd.\n", errCode%100, errCode/100)
		}
		if(pass) break; //done
		//After first-pass, use results of work-space query to allocate:
		lwork = int(work.data()[0]);
		work.resize(lwork);
	}
	logPrintf("done at t[s]: %.2lf.\n", clock_sec());
	
	//Get orthogonal matrix correspnding to Householder transformations:
	//--- initialize Q to identity:
	Buffer Q(nDataMine);
	double* Qptr = Q.data();
	for(int iCol: iColsMine)
		for(int iRow: iRowsMine)
			*(Qptr++) = (iRow==iCol ? 1. : 0.);
	work[0] = 0; lwork = -1; //for workspace query
	logPrintf("Extracting rotations ... "); logFlush();
	for(int pass=0; pass<2; pass++) //first pass is workspace query, next pass is actual calculation
	{	pdormhr_("Left", "NoTrans", &N, &N, &iLo, &iHi, H.data(), &one, &one, desc,
			tau.data(), Q.data(), &one, &one, desc, work.data(), &lwork, &info);
		if(info < 0)
		{	int errCode = -info;
			if(errCode < 100) die("Error in argument# %d to pdormhr.\n", errCode)
			else die("Error in entry %d of argument# %d to pdormhr.\n", errCode%100, errCode/100)
		}
		if(pass) break; //done
		//After first-pass, use results of work-space query to allocate:
		lwork = int(work.data()[0]);
		work.resize(lwork);
	}
	logPrintf("done at t[s]: %.2lf.\n", clock_sec());
	
	//Set H to strict upper Hessenberg form:
	double* Hdata = H.data();
	for(int iCol: iColsMine)
		for(int iRow: iRowsMine)
		{	if(iRow > iCol+1) *Hdata = 0.;
			Hdata++;
		}
	watch.stop();
	return Q;
}

extern "C" {
	//Tweaked version of pdhseqr to include intermediate progress (implemented in scalapack/PDHSEQRm.f)
	void pdhseqrm_(const char* job, const char* compz, const int* n, const int* ilo, const int* ihi, double* h, const int* desch,
		double* wr, double* wi, double* z, const int* descz, double* work, const int* lwork, int* iwork, const int* liwork, int* info);
	
	//For printing within the Fortran code: only use double precision printf formats (eg. %lf, %le, %lg)
	//and all required arguments in an array. If a format string is preceded by "t[s]: ", it will
	//be automatically susbtituted with the current execution time instead of an entry from args.
	//A newline will be added to the end of the print only if *endl is true.
	//IMPORTANT: the string fmt must end with a @ in Fortran since they are not null-terminated by default
	void logprint_(const char* fmt, const double* args, const bool* endl)
	{	const char* fmtEnd = fmt; while(*fmtEnd != '@') fmtEnd++;
		string buf(fmt, fmtEnd);
		int iArg = 0;
		while(buf.length())
		{	//Search till the next format specifier:
			auto iStart = buf.find_first_of("%");
			if(iStart == string::npos)
			{	//No more formatting required
				logPrintf(buf.c_str());
				buf.clear();
			}
			else
			{	//Find end of format string
				auto iStop = buf.find_first_of("aefg", iStart); //match any floating point format (case-insensitive)
				string token; //current piece to be processed
				if(iStop == string::npos)
					std::swap(token, buf); //last segment
				else
				{	token = buf.substr(0, iStop+1); //current segment
					buf = buf.substr(iStop+1, string::npos); //rest to be processed in next iteration
					double curArg = (token.find("t[s]:") == string::npos)
						? args[iArg++] //take next entry from args
						: clock_sec(); //substitute current time
					logPrintf(token.c_str(), curArg);
				}
			}
		}
		if(*endl) logPrintf("\n");
		logFlush();
	} 
}

//Schur decomposition and eigenvalues:
std::vector<complex> BlockCyclicMatrix::schur(Buffer& H, Buffer& Q, DiagMethod diagMethod) const
{	static StopWatch watch("BlockCyclicMatrix::schur"); watch.start();
	assert(H.size()==nDataMine);
	assert(Q.size()==nDataMine);
	assert((diagMethod==UsePDHSEQR) or (diagMethod==UsePDHSEQRm));
	auto pdhseqrFunc = (diagMethod==UsePDHSEQR) ? pdhseqr_ : pdhseqrm_;
	int iLo = 1, iHi = N;
	Buffer wr(N), wi(N); //real and imaginary parts of eigenvalues
	Buffer work(1); int lwork = -1, info = 0; //for workspace query
	std::vector<int> iwork(N); int liwork = -1; //for workspace query
	logPrintf("Schur decomposition ... "); logFlush();
	for(int pass=0; pass<2; pass++) //first pass is workspace query, next pass is actual calculation
	{	pdhseqrFunc("Schur", "Vectors", &N, &iLo, &iHi, H.data(), desc, wr.data(), wi.data(),
			Q.data(), desc, work.data(), &lwork, iwork.data(), &liwork, &info);
		if(info < 0)
		{	int errCode = -info;
			if(errCode < 100) die("Error in argument# %d to pdhseqr.\n", errCode)
			else die("Error in entry %d of argument# %d to pdhseqr.\n", errCode%100, errCode/100)
		}
		if(info > 0) die("Up to %d eigenvalues failed to converge.\n", info);
		if(pass) break; //done
		//After first-pass, use results of work-space query to allocate:
		lwork = liwork = 2*std::max(int(work.data()[0]), int(iwork.data()[0])); //note bug in pdhseqr: liwork underestimated
		work.resize(lwork);
		iwork.resize(liwork);
	}
	logPrintf("done at t[s]: %.2lf.\n", clock_sec());
	watch.stop();

	//Collect eigenvalues into complex array:
	std::vector<complex> evals(N);
	for(int i=0; i<N; i++)
		evals[i] = complex(wr[i], wi[i]);
	return evals;
}

//Eigenvector transformation
void BlockCyclicMatrix::getEvecs(const Buffer& T, const Buffer& Q, Buffer& VR, Buffer& VL, const Buffer* scaleFactors) const
{	static StopWatch watchLeft("BlockCyclicMatrix::leftEvecs"), watchRight("BlockCyclicMatrix::rightEvecs");
	assert(T.size()==nDataMine);
	assert(Q.size()==nDataMine);
	if(scaleFactors) assert(int(scaleFactors->size())==N);
	
	//Underflow/overflow and precision constants:
	double uFlow = dlamch_("Safe minimum");
	double oFlow = 1./uFlow;
	dlabad_(&uFlow, &oFlow);
	const double prec = dlamch_("Precision");
	const double sNum = uFlow*(N/prec);
	
	//Collect column 1-norms and tri-diagonal portion of matrix on all processes:
	Buffer tNorm(N, 0.); //column 1-norms used for overflow mitigation below
	Buffer tDiag(N, 0.), tDiagU(N, 0.), tDiagL(N, 0.); //diagonal, upper and lower diagonal entries
	{	const double* Tdata = T.data();
		for(int j: iColsMine)
			for(int i: iRowsMine)
			{	const double& t = *(Tdata++);
				if(t) tNorm[j] += fabs(t);
				if(i == j) tDiag[i] = t;
				if(i+1 == j) tDiagU[i] = t;
				if(j+1 == i) tDiagL[j] = t;
			}
	}
	mpiUtil->allReduceData(tNorm, MPIUtil::ReduceSum);
	mpiUtil->allReduceData(tDiag, MPIUtil::ReduceSum);
	mpiUtil->allReduceData(tDiagU, MPIUtil::ReduceSum);
	mpiUtil->allReduceData(tDiagL, MPIUtil::ReduceSum);
	
	//Left eigenvector calculation:
	watchLeft.start();
	logPrintf("Computing left eigenvectors of Schur matrix ... "); logFlush();
	const int one = 1, two = 2, notTrans = 0; int info = 0;
	double zeroD = 0., oneD = 1.;
	double rhs[4], x[4], xNorm, scale; //1x1 or 2x2 matrices used in dlanl2; norm and scale factor of x
	//--- initialize Z to transpose(T):
	Buffer Z(nDataMine); //left eigenvectors of T (multiplied by Q at the end)
	pdgeadd_("T", &N, &N, &oneD, T.data(),&one,&one,desc, &zeroD, Z.data(),&one,&one,desc);
	//--- compute scale factors for initial RHS of the subsequent triangular solves
	//--- in the process, also zero-out block-diagonal part of transpose(T) stored in Z 
	//--- and prepare the indexing arrays required for synchonizing 2x2 blocks split across processes
	Buffer ZdiagMine; ZdiagMine.reserve(nColsMine);
	int iProcRowNext = positiveRemainder(iProcRow+1, nProcsRow), iProcRowPrev = positiveRemainder(iProcRow-1, nProcsRow);
	int iProcColNext = positiveRemainder(iProcCol+1, nProcsCol), iProcColPrev = positiveRemainder(iProcCol-1, nProcsCol);
	std::vector<int> iColsMinePadded; //iColMine whose 2x2 operations will happen on this process, and -1 where data is needed from iProcColNext
	std::vector<int> iPaddedNext; //indices into iColsMinePadded for data that should be received from, processed and sent back to iProcColNext
	std::vector<int> iColsMinePrev; //iColMine that needs to be sent to iProcColPrev, processed there and then received back
	int nBlocksPerCol = ceildiv(nColsMine,blockSize);
	iColsMinePadded.reserve(nColsMine + nBlocksPerCol);
	iPaddedNext.reserve(nBlocksPerCol);
	iColsMinePrev.reserve(nBlocksPerCol);
	for(int ki=0; ki<N; ki++)
	{	bool complexPair = (ki+1<N) and (tDiagL[ki]!=0.);
		int kiBlockSize = complexPair ? 2  : 1; //current block size in ki
		int kiStop = ki + kiBlockSize-1; //end of current block in ki
		int iRowMine, iColMine;
		#define IF_COL_MINE(iCol) iColMine = localColIndex(iCol); if(iColMine >= 0)
		#define ZERO_ENTRY_Z(iRow) iRowMine = localRowIndex(iRow); if(iRowMine >= 0) Z[iRowMine+iColMine*nRowsMine] = 0.;
		if(complexPair)
		{	double wi = sqrt(fabs(tDiagL[ki] * tDiagU[ki]));
			bool Ugreater = (fabs(tDiagU[ki]) > fabs(tDiagL[ki]));
			IF_COL_MINE(ki)     { ZdiagMine.push_back(Ugreater ? wi/tDiagU[ki] : 1.);  ZERO_ENTRY_Z(ki) ZERO_ENTRY_Z(kiStop) }
			IF_COL_MINE(kiStop) { ZdiagMine.push_back(Ugreater ? 1. : -wi/tDiagL[ki]); ZERO_ENTRY_Z(ki) ZERO_ENTRY_Z(kiStop) }
			//Update 2x2 sync arrays:
			iColMine = localColIndex(ki); int iColMine2 = localColIndex(kiStop);
			if(iColMine >= 0)
			{	if(iColMine2 >= 0)
				{	//both ki and kiStop on this process; no communication needed
					iColsMinePadded.push_back(iColMine);
					iColsMinePadded.push_back(iColMine2);
				}
				else
				{	//ki on this process, but kiStop on next process 
					iColsMinePadded.push_back(iColMine);
					iColsMinePadded.push_back(-1); //need to recv kiStop from next process
					iPaddedNext.push_back(iColsMinePadded.size()-1); //point to last entry added above
				}
			}
			else
			{	if(iColMine2 >= 0)
				{	//kiStop on this process, but ki on prev process
					iColsMinePrev.push_back(iColMine2); //send data to prev process, which will process the 2x2 block
				}
				//else both not ki and kiStop not involved in this process
			}
		}
		else
		{	IF_COL_MINE(ki) { ZdiagMine.push_back(1.); ZERO_ENTRY_Z(ki) } 
			iColMine = localColIndex(ki);
			if(iColMine >= 0) iColsMinePadded.push_back(iColMine); //never need communictaion for 1x1 block
		}
		#undef IF_COL_MINE
		#undef ZERO_ENTRY_Z
		ki = kiStop;
	}
	//--- apply diagonal scaling factors
	//--- (diagonal blocks still left at zero, so Z is strictly lower triangular)
	{	double* Zdata = Z.data();
		for(int iColMine=0; iColMine<nColsMine; iColMine++)
		{	cblas_dscal(nRowsMine, -ZdiagMine[iColMine], Zdata,1);
			Zdata += nRowsMine;
		}
	}
	//--- solve set of quasi-triangular systems (T - (wr-i*wi))*x = Z in parallel
	Buffer Tcur(nRowsMine*2), Zupdate(nColsMine*2);
	int nColsPadded = iColsMinePadded.size();
	int nColsPrev = iColsMinePrev.size();
	int nColsNext = iPaddedNext.size();
	Buffer xMine(nColsPadded*2); //buffer used for the kiBlockSize x jBlockSize updates
	Buffer prevBuf(nColsPrev), nextBuf(nColsNext); //buffers for sending/recv'ing from prev and next process
	for(int j=0; j<N; j++)
	{	int jBlockSize = ((j+1<N) and (tDiagL[j]!=0.)) ? 2 : 1; //current block size in j
		int jStop = j+jBlockSize-1; //end of current block in j
		//Determine data ranges requiring opration for these j:
		int iColMineStop, iRowMineStop, iUnusedStart;
		getRange(iColsMine, 0, j, iUnusedStart, iColMineStop); //Note that iColMineStart = 0 
		getRange(iRowsMine, 0, j, iUnusedStart, iRowMineStop); //Note that iRowMineStart = 0 
		//Make T[:,j] available on all processes in each process row
		if(iRowMineStop)
			for(int bj=0; bj<jBlockSize; bj++)
			{	int whoseProcCol = ((j+bj) / blockSize) % nProcsCol;
				if(whoseProcCol == iProcCol)
					eblas_copy(&Tcur[nRowsMine*bj], &T[nRowsMine*localColIndex(j+bj)], iRowMineStop);
				mpiRow->bcast(&Tcur[nRowsMine*bj], iRowMineStop, whoseProcCol);
			}
		//Update  Z(j,ki) -= sum_(ki < i < j) [ T(i,j) * Z(i,ki) ]
		//--- done as  Z(j,ki) -= sum_(i < j) [ T(i,j) * Z(i,ki) ] since Z is strictly lower triangular (diag part set later)
		if(iColMineStop)
		{	//Compute the local piece of the matrix product
			if(iRowMineStop)
				cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, iColMineStop, jBlockSize, iRowMineStop,
					1., Z.data(),nRowsMine, Tcur.data(),nRowsMine, 0., Zupdate.data(),nColsMine);
			//Accumulate result on the appropriate process
			for(int bj=0; bj<jBlockSize; bj++)
			{	int whoseProcRow = ((j+bj) / blockSize) % nProcsRow;
				mpiCol->reduce(&Zupdate[nColsMine*bj], iColMineStop, MPIUtil::ReduceSum, whoseProcRow);
				if(whoseProcRow == iProcRow)
					cblas_daxpy(iColMineStop, -1., &Zupdate[nColsMine*bj],1, &Z[localRowIndex(j+bj)],nRowsMine);
			}
		}
		//Solve kiBlockSize x jBlockSize complex equations to update Z:
		//--- prepare RHS, accounting for blocks split across processes if any
		bool localRow[2];
		for(int bj=0; bj<jBlockSize; bj++)
		{	int iRowMine = localRowIndex(j+bj);
			localRow[bj] = (iRowMine >= 0);
			if(localRow[bj])
			{	//Send columns of Z to prev process:
				MPIUtil::Request sendRequest;
				if(nColsPrev)
				{	for(int i=0; i<nColsPrev; i++)
						prevBuf[i] = Z[iRowMine+iColsMinePrev[i]*nRowsMine];
					mpiRow->sendData(prevBuf, iProcColPrev, j+bj, &sendRequest);
				}
				//Recv columns of Z from next process:
				if(nColsNext)
				{	mpiRow->recvData(nextBuf, iProcColNext, j+bj);
					for(int i=0; i<nColsNext; i++)
						xMine[iPaddedNext[i]+bj*nColsPadded] = nextBuf[i];
				}
				//Collect local pieces:
				for(int i=0; i<nColsPadded; i++)
				{	int iColMine = iColsMinePadded[i];
					if(iColMine >= 0)
						xMine[i+bj*nColsPadded] = Z[iRowMine+iColMine*nRowsMine];
				}
				if(nColsPrev) mpiRow->wait(sendRequest);
			}
		}
		if(jBlockSize == 2)
		{	if(localRow[0] and (not localRow[1])) mpiCol->recv(&xMine[nColsPadded], nColsPadded, iProcRowNext, j+1); //recv second row from next process
			if(localRow[1] and (not localRow[0])) mpiCol->send(&xMine[nColsPadded], nColsPadded, iProcRowPrev, j+1); //send second row to prev process
		}
		//--- perform kiBlockSize x jBlockSize updates from rhsMine to xMine
		if(localRow[0])
		{	const double T22[4] = { tDiag[j], tDiagU[j], tDiagL[j], tDiag[jStop] }; //2x2 diagonal block of T (only 1x1 valid/needed if jStop==j)
			for(int iPadded=0; iPadded<nColsPadded; iPadded++)
			{	int iColMine = iColsMinePadded[iPadded];
				assert(iColMine >= 0);
				int ki = iColsMine[iColMine];
				if(ki >= j) break;
				bool complexPair = (ki+1<N) and (tDiagL[ki]!=0.);
				int kiBlockSize = complexPair ? 2  : 1; //current block size in ki
				int iPaddedStop = iPadded + kiBlockSize-1; //end of current block in iPadded
				double wr = tDiag[ki];
				double mwi = complexPair ? -sqrt(fabs(tDiagL[ki]*tDiagU[ki])) : 0;
				double sMin = std::max(prec*(fabs(wr)+fabs(mwi)), sNum); //small number threshold for this eigenvector (pair)
				//Fetch RHS:
				for(int bk=0; bk<kiBlockSize; bk++)
					for(int bj=0; bj<jBlockSize; bj++)
						rhs[bj+2*bk] = xMine[(iPadded+bk)+bj*nColsPadded];
				//Perform block solve:
				dlaln2_(&notTrans, &jBlockSize, &kiBlockSize, &sMin, &oneD,
					T22, &two, &oneD, &oneD, 
					rhs, &two, &wr, &mwi, x, &two,
					&scale, &xNorm, &info);
				if(scale != 1.) die_alone("Overflow encountered.\n");
				//Set x back:
				for(int bk=0; bk<kiBlockSize; bk++)
					for(int bj=0; bj<jBlockSize; bj++)
						xMine[(iPadded+bk)+bj*nColsPadded] = x[bj+2*bk];
				iPadded = iPaddedStop;
			}
		}
		//--- set results from xMine back to Z, accounting for blocks split across processes if any
		if(jBlockSize == 2)
		{	if(localRow[0] and (not localRow[1])) mpiCol->send(&xMine[nColsPadded], nColsPadded, iProcRowNext, j+1); //send second row back to next process
			if(localRow[1] and (not localRow[0])) mpiCol->recv(&xMine[nColsPadded], nColsPadded, iProcRowPrev, j+1); //recv second row back from prev process
		}
		for(int bj=0; bj<jBlockSize; bj++)
			if(localRow[bj])
			{	int iRowMine = localRowIndex(j+bj);
				//Send columns of Z back to next process:
				MPIUtil::Request sendRequest;
				if(nColsNext)
				{	for(int i=0; i<nColsNext; i++)
						nextBuf[i] = xMine[iPaddedNext[i]+bj*nColsPadded];
					mpiRow->sendData(nextBuf, iProcColNext, j+bj, &sendRequest);
				}
				//Recv columns of Z back from prev process:
				if(nColsPrev)
				{	mpiRow->recvData(prevBuf, iProcColPrev, j+bj);
					for(int i=0; i<nColsPrev; i++)
						Z[iRowMine+iColsMinePrev[i]*nRowsMine] = prevBuf[i];
				}
				//Set local pieces back in Z:
				for(int i=0; i<nColsPadded; i++)
				{	int iColMine = iColsMinePadded[i];
					if(iColMine >= 0)
						Z[iRowMine+iColMine*nRowsMine] = xMine[i+bj*nColsPadded];
				}
				if(nColsNext) mpiRow->wait(sendRequest);
			}
		j = jStop;
	}
	//--- set the diagonal blocks of Z:
	for(int iColMine=0; iColMine<nColsMine; iColMine++)
	{	int iRowMine = localRowIndex(iColsMine[iColMine]);
		if(iRowMine >= 0)
			Z[iRowMine+iColMine*nRowsMine] = ZdiagMine[iColMine];
	}
	logPrintf("done at t[s]: %.2lf.\n", clock_sec());
	//--- multiply by Q
	logPrintf("Rotating left eigenvectors to original basis ... "); logFlush();
	matMult(1., Q,false, Z,false, 0.,VL);
	//--- account for scaleFactors if necessary:
	if(scaleFactors)
	{	//Collect scale factors relevant to my rows:
		Buffer scaleMineInv; scaleMineInv.reserve(nRowsMine);
		for(int iRow: iRowsMine)
			scaleMineInv.push_back(1./scaleFactors->at(iRow));
		//Apply scale factors:
		double* VLdata = VL.data();
		for(int iColMine=0; iColMine<nColsMine; iColMine++)
			for(int iRowMine=0; iRowMine<nRowsMine; iRowMine++)
				*(VLdata++) *= scaleMineInv[iRowMine];
	}
	logPrintf("done at t[s]: %.2lf.\n", clock_sec());
	watchLeft.stop();
	
	//Right eigenvector calculation:
	watchRight.start();
	logPrintf("Computing right eigenvectors ... "); logFlush();
	//--- create transpose(VL) as an LHS matrix for inversion:
	Buffer VLT(nDataMine);
	pdgeadd_("T", &N, &N, &oneD, VL.data(),&one,&one,desc, &zeroD, VLT.data(),&one,&one,desc);
	//--- set VR to identity:
	VR.resize(nDataMine);
	double* VRdata = VR.data();
	for(int iCol: iColsMine)
		for(int iRow: iRowsMine)
			*(VRdata++) = (iRow==iCol) ? 1. : 0.;
	//--- update VR = inv(transpose(VL))
	info = 0;
	std::vector<int> pivot(N);
	pdgesv_(&N, &N, VLT.data(), &one, &one, desc, pivot.data(), VR.data(), &one, &one, desc, &info);
	if(info < 0)
	{	int errCode = -info;
		if(errCode < 100) die("Error in argument# %d to pdgesv.\n", errCode)
		else die("Error in entry %d of argument# %d to pdgesv.\n", errCode%100, errCode/100)
	}
	if(info > 0) die("Matrix singular at column# %d in pdgesv.\n", info);
	logPrintf("done at t[s]: %.2lf.\n", clock_sec());
	watchRight.stop();
}

void BlockCyclicMatrix::matMult(double alpha, const Buffer& A, bool transA, const Buffer& B, bool transB, double beta, Buffer& C) const
{	static StopWatch watch("BlockCyclicMatrix::matMult"); watch.start();
	assert(A.size()==nDataMine);
	assert(B.size()==nDataMine);
	if(beta) assert(C.size()==nDataMine); else C.resize(nDataMine);
	char transAchar = transA ? 'T' : 'N';
	char transBchar = transB ? 'T' : 'N';
	int one = 1;
	pdgemm_(&transAchar, &transBchar, &N, &N, &N, &alpha,
		A.data(), &one, &one, desc,
		B.data(), &one, &one, desc, &beta,
		C.data(), &one, &one, desc);
	watch.stop();
}

void BlockCyclicMatrix::matMultVec(double alpha, const Buffer& A, const Buffer& B, Buffer& C) const
{	static StopWatch watch("BlockCyclicMatrix::matMultVec"); watch.start();
	assert(A.size()==nDataMine);
	assert(B.size() % nRowsMine == 0);
	int nVec = B.size() / nRowsMine;
	C.resize(nVec * nColsMine);
	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, nColsMine, nVec, nRowsMine,
		alpha, A.data(), nRowsMine, B.data(), nRowsMine, 0., C.data(), nColsMine);
	mpiCol->allReduceData(C, MPIUtil::ReduceSum);
	watch.stop();
}

#endif //SCALAPACK_ENABLED
