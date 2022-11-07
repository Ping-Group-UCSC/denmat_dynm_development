/*-------------------------------------------------------------------
Copyright 2018 Ravishankar Sundararaman

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

#include "DistributedMatrix.h"
#include <fftw3-mpi.h>
#include <algorithm>

template<typename scalar> void readMatrix(const MPIUtil* mpiUtil, const std::shared_ptr<MPIUtil> mpiInterGroup,
	string fname, int nElemsTot, int nElems, int iElemStart, int nCellsTot, complex* dest)
{
	logPrintf("Reading '%s' ... ", fname.c_str()); fflush(globalLog);
	size_t fsizeExpected = nCellsTot * nElemsTot * sizeof(scalar);
	if(!mpiInterGroup || mpiInterGroup->isHead()) //read everywhere if no intergroup MPI, or only on head of inter-group communicator
	{	MPIUtil::File fp;
		mpiUtil->fopenRead(fp, fname.c_str(), fsizeExpected);
		std::vector<scalar> column(nElems); //temporary storage to hold local portion of column
		for(int iCell=0; iCell<nCellsTot; iCell++)
		{	mpiUtil->fseek(fp, (iCell*nElemsTot+iElemStart)*sizeof(scalar), SEEK_SET);
			mpiUtil->fread(column.data(), sizeof(scalar), nElems, fp);
			for(int iElem=0; iElem<nElems; iElem++)
				dest[iCell+nCellsTot*iElem] = column[iElem]; //store transposed
		}
		mpiUtil->fclose(fp);
	}
	if(mpiInterGroup) //broadcast within each inter-group communictaor
		mpiInterGroup->bcast<complex>(dest, nCellsTot*nElems);
	logPrintf("done.\n");
}

struct PlanSet
{	fftw_plan transpose; //plan for MPI transpose
	fftw_plan fft; //Fourier transform plan for non-squared case
	fftw_plan fft1; //Fourier transform plan w.r.t iR1 for squared case
	fftw_plan fft2; //Fourier transform plan w.r.t iR2 for squared case
};

DistributedMatrix::DistributedMatrix(string fname, bool realOnly, const MPIUtil* mpiUtil, int nElemsTot,
	const std::vector<vector3<int>>& cellMap, const vector3<int>& kfold, bool squared,
	const std::shared_ptr<MPIUtil>  mpiInterGroup, const std::vector<matrix>* cellWeights, const vector3<int>* kfoldInPtr,
	int derivDir, const DistributedMatrix* derivData, const matrix3<>* R)
: mpiUtil(mpiUtil), nElemsTot(nElemsTot), cellMap(cellMap), kfold(kfold), squared(squared), deriv(derivDir >= 0)
{
	if(squared) assert(cellWeights); //only unique cells supported in square dmode (e-ph elements)
	if(deriv)
	{	assert(not squared); //derivative only supported for non-squared mode
		assert(R); //need lattice vectors for Caretsian conversion
		derivRi = R->row(derivDir);
	}
	kfoldProd = kfold[0]*kfold[1]*kfold[2];
	kfoldIn = kfold;
	if(kfoldInPtr)
	{	assert(not squared); //different input mesh dimension only supported in single k (non-squared) mode
		kfoldIn = *kfoldInPtr;
	}
	kfoldInProd = kfoldIn[0]*kfoldIn[1]*kfoldIn[2];
	nCellsTot = cellWeights ? kfoldInProd : cellMap.size(); //cell weights optionally applied here to save disk / memory
	nkTot = kfoldProd;
	if(squared)
	{	nCellsTot *= nCellsTot;
		nkTot *= nkTot;
	}
	
	//Determine division:
	ptrdiff_t local_n0, local_0start, local_n1, local_1start;
	ptrdiff_t nTot = fftw_mpi_local_size_2d_transposed(nElemsTot, nkTot,
		mpiUtil->communicator(), &local_n0, &local_0start, &local_n1, &local_1start);
	nElems = local_n0; iElemStart = local_0start;
	nk = local_n1; ikStart = local_1start;
	
	//Make ikStart for all processes available:
	ikStartProc.resize(mpiUtil->nProcesses()+1);
	ikStartProc[0] = 0;
	ikStartProc[mpiUtil->iProcess()+1] = ikStart + nk;
	for(int jProc=0; jProc<mpiUtil->nProcesses(); jProc++)
	{	if(!nk && mpiUtil->iProcess()==jProc)
			ikStart = ikStartProc[jProc+1] = ikStartProc[jProc]; //because FFTW doesn't set the ikStart if nk=0
		mpiUtil->bcast(ikStartProc[jProc+1], jProc);
	}
	ikStartProc[mpiUtil->nProcesses()] = nkTot;
	assert(ikStartProc[mpiUtil->iProcess()+1] == ikStart+nk);
	
	//Make iElemStart for all processes available:
	iElemStartProc.resize(mpiUtil->nProcesses()+1);
	iElemStartProc[0] = 0;
	iElemStartProc[mpiUtil->iProcess()+1] = iElemStart + nElems;
	for(int jProc=0; jProc<mpiUtil->nProcesses(); jProc++)
	{	if(!nElems && mpiUtil->iProcess()==jProc)
			iElemStart = iElemStartProc[jProc+1] = iElemStartProc[jProc]; //because FFTW doesn't set the iElemStart if nk=0
		mpiUtil->bcast(iElemStartProc[jProc+1], jProc);
	}
	iElemStartProc[mpiUtil->nProcesses()] = nElemsTot;
	assert(iElemStartProc[mpiUtil->iProcess()+1] == iElemStart+nElems);
	
	//Allocate matrix and buffer:
	mat.init(nElems*nCellsTot);
	buf.init(nTot); //nTot = max(nElems*nkTot, nElemsTot*nk)
	
	//Read / copy matrix data:
	if((derivDir >= 0) and derivData)
	{	//Copy data:
		assert(mpiUtil == derivData->mpiUtil);
		assert(nElemsTot == derivData->nElemsTot);
		assert(nCellsTot == derivData->nCellsTot);
		mat = derivData->mat;
	}
	else
	{	//Read from file
		if(realOnly) readMatrix<double>(mpiUtil, mpiInterGroup, fname, nElemsTot, nElems, iElemStart, nCellsTot, mat.data());
		else readMatrix<complex>(mpiUtil, mpiInterGroup, fname, nElemsTot, nElems, iElemStart, nCellsTot, mat.data());
	}
	
	//Create FFTW plans:
	complex* bufData = buf.data();
	planSet = std::make_shared<PlanSet>();
	//--- transpose
	planSet->transpose = fftw_mpi_plan_many_transpose(nElemsTot, nkTot, 2,
		FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
		(double*)bufData, (double*)bufData,
		mpiUtil->communicator(), FFTW_MEASURE);
	if(!planSet->transpose) die_alone("MPI transpose plan creation failed.\n");
	//--- FFTs:
	if(squared)
	{	planSet->fft1 = fftw_plan_many_dft(3, &kfold[0], kfoldProd, //this has to be called nElems times
			(fftw_complex*)bufData, NULL, kfoldProd, 1, //Note: strided transform
			(fftw_complex*)bufData, NULL, kfoldProd, 1,
			-1, FFTW_MEASURE);
		if(!planSet->fft1) die_alone("Cell-map-squared FFT w.r.t cell 1 plan creation failed.\n");
		planSet->fft2 = fftw_plan_many_dft(3, &kfold[0], nElems*kfoldProd,
			(fftw_complex*)bufData, NULL, 1, kfoldProd,
			(fftw_complex*)bufData, NULL, 1, kfoldProd,
			+1, FFTW_MEASURE);
		if(!planSet->fft2) die_alone("Cell-map-squared FFT w.r.t cell 2 plan creation failed.\n");
	}
	else
	{	planSet->fft = fftw_plan_many_dft(3, &kfold[0], nElems,
			(fftw_complex*)bufData, NULL, 1, kfoldProd,
			(fftw_complex*)bufData, NULL, 1, kfoldProd,
			+1, FFTW_MEASURE);
		if(!planSet->fft) die_alone("Cell-map FFT plan creation failed.\n");
	}
	
	//Initialize cell index:
	if(cellWeights)
	{	uniqueCells.resize(kfoldProd);
		for(size_t iCell=0; iCell<cellMap.size(); iCell++)
		{	Cell cell;
			cell.iR = cellMap[iCell];
			cell.indexIn = calculateIndex(cell.iR, kfoldIn);
			//Get real part of matrix:
			const matrix& w = cellWeights->at(iCell);
			cell.weight.reserve(w.nData());
			if(squared)
			{	nAtoms = w.nRows();
				nBands = w.nCols();
				for(int iAtom=0; iAtom<nAtoms; iAtom++)
					for(int iBand=0; iBand<nBands; iBand++)
						cell.weight.push_back(w(iAtom, iBand).real());
			}
			else
			{	nAtoms = 0;
				nBands = w.nRows();
				assert(w.nRows() == w.nCols());
				cell.weight.assign(w.nData(), 0.);
				eblas_daxpy(w.nData(), 1., (const double*)w.data(),2, cell.weight.data(),1); //get real parts
			}
			uniqueCells[calculateIndex(cell.iR, kfold)].push_back(cell);
		}
		for(std::vector<Cell>& cells: uniqueCells)
			if(cells.size() > 1)
				std::sort(cells.begin(), cells.end());
		nModesPerAtom = squared ? (nElemsTot/(nBands*nBands*nAtoms)) : 0;
	}
	else
	{	cellIndex.resize(nCellsTot);
		auto iter = cellIndex.begin();
		for(const vector3<int>& iR: cellMap)
			*(iter++) = calculateIndex(iR, kfold);
		assert(iter == cellIndex.end());
	}
}

DistributedMatrix::~DistributedMatrix()
{	//Clean-up FFTW plans:
	fftw_destroy_plan(planSet->transpose);
	if(squared)
	{	fftw_destroy_plan(planSet->fft1);
		fftw_destroy_plan(planSet->fft2);
	}
	else fftw_destroy_plan(planSet->fft);
}

void DistributedMatrix::transform(vector3<> k0)
{	static StopWatch watch("DistributedMatrix::transform1"); watch.start();
	assert(!squared);
	if(uniqueCells.size()) //Unique cell mode
	{	initializePhase(k0);
		//Apply cell weights and offset phases:
		int matStride = nBands*nBands; //number of elements per matrix
		int iMatStart = iElemStart / matStride;
		int iMatStop = ceildiv(iElemStart+nElems, matStride);
		complex* bufData = buf.data();
		const complex* matData = mat.data();
		for(int iMat=iMatStart; iMat<iMatStop; iMat++)
		{	//Determine index range of w that contributes
			int iElemOffset = iMat*matStride - iElemStart;
			int iwStart = std::max(-iElemOffset, 0);
			int iwStop = std::min(nElems-iElemOffset, matStride);
			for(int iw=iwStart; iw<iwStop; iw++)
			{	for(int iCell=0; iCell<kfoldProd; iCell++)
				{	complex result; //current result
					const std::vector<Cell>& cells = uniqueCells[iCell];
					complex wPhase; //current weight * phase sum for unique cells
					for(auto iter=cells.begin(); iter!=cells.end();)
					{	auto iterNext = iter; iterNext++;
						wPhase += iter->phase01 * iter->weight[iw];
						if(iterNext==cells.end() or iterNext->indexIn!=iter->indexIn)
						{	result += wPhase * matData[iter->indexIn];
							wPhase = complex();
						}
						iter = iterNext;
					}
					*(bufData++) = result;
				}
				matData += nCellsTot;
			}
		}
	}
	else //Full cellMap mode:
	{	std::vector<complex> phase0;
		initializePhase(k0, phase0);
		//Reduce from mat to buf (apply offset phases and combine equivalent cells):
		buf.zero();
		complex* bufData = buf.data();
		const complex* matData = mat.data();
		for(int iElem=0; iElem<nElems; iElem++)
		{	auto cellIndexPtr = cellIndex.begin();
			for(const complex& phase0cur: phase0)
				bufData[iElem*nkTot+*(cellIndexPtr++)] += *(matData++) * phase0cur;
		}
	}
	//Apply Fourier transform followed by MPI transpose:
	fftw_execute(planSet->fft);
	fftw_execute(planSet->transpose);
	watch.stop();
}

void DistributedMatrix::transform(vector3<> k01, vector3<> k02)
{	static StopWatch watch("DistributedMatrix::transform2"); watch.start();
	assert(squared);
	//Initialize offset phases:
	for(std::vector<Cell>& cells: uniqueCells)
		for(Cell& cell: cells)
		{	cell.phase01 = cis(-2*M_PI*dot(cell.iR, k01));
			cell.phase02 = cis(+2*M_PI*dot(cell.iR, k02));
		}
	//Copy mat to buf:
	callPref(eblas_copy)(buf.data(), mat.data(), mat.nData());
	//Apply cell weights and offset phases:
	int atomStride = nModesPerAtom*nBands*nBands; //number of elements per atom
	int iAtomStart = iElemStart / atomStride;
	int iAtomStop = ceildiv(iElemStart+nElems, atomStride);
	int nBandsSq = nBands*nBands;
	complex* bufData = buf.data();
	for(int iAtom=iAtomStart; iAtom<iAtomStop; iAtom++)
		for(int iVector=0; iVector<nModesPerAtom; iVector++)
		{	int iElemOffset = (nModesPerAtom*iAtom+iVector)*nBandsSq - iElemStart;
			int iwStart = std::max(-iElemOffset, 0);
			int iwStop = std::min(nElems-iElemOffset, nBandsSq);
			if(iwStop <= iwStart) continue; //nothing on current process
			//Loop over band pairs:
			int iw = iwStart;
			int b2 = iw/nBands;
			int b1 = iw - b2*nBands;
			std::vector<complex> w1(kfoldProd), w2(kfoldProd); //precalculated weights
			#define CALC_w(j) \
				for(int iCell=0; iCell<kfoldProd; iCell++) \
				{	w##j[iCell] = 0.; \
					for(const Cell& c: uniqueCells[iCell]) \
						w##j[iCell] += c.phase0##j * c.weight[iAtom*nBands + b##j]; \
				}
			CALC_w(1)
			CALC_w(2)
			while(iw<iwStop)
			{
				//Apply weights:
				for(int iCell1=0; iCell1<kfoldProd; iCell1++)
					for(int iCell2=0; iCell2<kfoldProd; iCell2++)
						*(bufData++) *= (w1[iCell1] * w2[iCell2]);
				
				iw++; if(iw==iwStop) break;
				b1++;
				if(b1==nBands)
				{	b1=0;
					b2++;
					CALC_w(2) //update w2
				}
				CALC_w(1) //update w1
			}
			#undef CALC_w
		}
	//Apply Fourier transform followed by MPI transpose:
	bufData = buf.data();
	for(int iElem=0; iElem<nElems; iElem++)
	{	fftw_execute_dft(planSet->fft1, (fftw_complex*)bufData, (fftw_complex*)bufData);
		bufData += nkTot;
	}
	fftw_execute(planSet->fft2);
	fftw_execute(planSet->transpose);
	watch.stop();
}

const complex* DistributedMatrix::getResult(int ik) const
{	int ikLocal = ik-ikStart;
	assert(ikLocal >= 0);
	assert(ikLocal < nk);
	return buf.data() + ikLocal*nElemsTot;
}


void DistributedMatrix::compute(vector3<> k)
{	static StopWatch watch("DistributedMatrix::compute1"); watch.start();
	assert(!squared);
	if(uniqueCells.size()) //Unique cell mode
	{	initializePhase(k);
		//Discrete Fourier transform for k:
		const complex* matData = mat.data();
		complex* bufData = buf.data();
		int matStride = nBands*nBands; //number of elements per matrix
		int iMatStart = iElemStart / matStride;
		int iMatStop = ceildiv(iElemStart+nElems, matStride);
		for(int iMat=iMatStart; iMat<iMatStop; iMat++)
		{	//Determine index range of weight matrix that contributes
			int iElemOffset = iMat*matStride - iElemStart;
			int iwStart = std::max(-iElemOffset, 0);
			int iwStop = std::min(nElems-iElemOffset, matStride);
			for(int iw=iwStart; iw<iwStop; iw++)
			{	complex out = 0.;
				for(int iCell=0; iCell<kfoldProd; iCell++)
				{	const std::vector<Cell>& cells = uniqueCells[iCell];
					complex wPhase; //current weight * phase sum for unique cells
					for(auto iter=cells.begin(); iter!=cells.end();)
					{	auto iterNext = iter; iterNext++;
						wPhase += iter->phase01 * iter->weight[iw];
						if(iterNext==cells.end() or iterNext->indexIn!=iter->indexIn)
						{	out += wPhase * matData[iter->indexIn];
							wPhase = complex();
						}
						iter = iterNext;
					}
				}
				*(bufData++) = out;
				matData += nCellsTot;
			}
		}
	}
	else //Full cellMap mode:
	{	std::vector<complex> phase;
		initializePhase(k, phase);
		//Discrete Fourier transform for k (as a matrix-vector multiply):
		const complex one = 1., zero = 0.;
		cblas_zgemv(CblasColMajor, CblasTrans,
			nCellsTot, nElems, &one, mat.data(), nCellsTot,
			phase.data(), 1,
			&zero, buf.data(), 1);
	}
	//Collect data on head:
	collectProc();
	watch.stop();
}

void DistributedMatrix::compute(vector3<> k1, vector3<> k2, int ik, int iProc)
{	static StopWatch watch("DistributedMatrix::compute2"); watch.start();
	assert(squared);
	//Initialize phases:
	for(std::vector<Cell>& cells: uniqueCells)
		for(Cell& cell: cells)
		{	cell.phase01 = cis(-2*M_PI*dot(cell.iR, k1));
			cell.phase02 = cis(+2*M_PI*dot(cell.iR, k2));
		}
	//Discrete Fourier transform over k1 and k2:
	ManagedArray<complex> bufTmp; bufTmp.init(std::max(1, nElems));
	const complex* matData = mat.data();
	complex* bufData = bufTmp.data(); //work with a temporary buffer before collecting on one process
	int atomStride = nModesPerAtom*nBands*nBands; //number of elements per atom
	int iAtomStart = iElemStart / atomStride;
	int iAtomStop = ceildiv(iElemStart+nElems, atomStride);
	int nBandsSq = nBands*nBands;
	for(int iAtom=iAtomStart; iAtom<iAtomStop; iAtom++)
		for(int iVector=0; iVector<nModesPerAtom; iVector++)
		{	int iElemOffset = (nModesPerAtom*iAtom+iVector)*nBandsSq - iElemStart;
			int iwStart = std::max(-iElemOffset, 0);
			int iwStop = std::min(nElems-iElemOffset, nBandsSq);
			if(iwStop <= iwStart) continue; //nothing on current process
			//Loop over band pairs:
			int iw = iwStart;
			int b2 = iw/nBands;
			int b1 = iw - b2*nBands;
			std::vector<complex> w1(kfoldProd), w2(kfoldProd); //precalculated weights
			#define CALC_w(j) \
				for(int iCell=0; iCell<kfoldProd; iCell++) \
				{	w##j[iCell] = 0.; \
					for(const Cell& c: uniqueCells[iCell]) \
						w##j[iCell] += c.phase0##j * c.weight[iAtom*nBands + b##j]; \
				}
			CALC_w(1)
			CALC_w(2)
			while(iw<iwStop)
			{
				//Apply weights:
				complex out = 0;
				for(int iCell1=0; iCell1<kfoldProd; iCell1++)
					for(int iCell2=0; iCell2<kfoldProd; iCell2++)
						out += (w1[iCell1] * w2[iCell2]) * (*(matData++));
				*(bufData++) = out;
				
				iw++; if(iw==iwStop) break;
				b1++;
				if(b1==nBands)
				{	b1=0;
					b2++;
					CALC_w(2) //update w2
				}
				CALC_w(1) //update w1
			}
			#undef CALC_w
		}
	//Collect data on head:
	collectProc(bufTmp.data(), ik, iProc);
	watch.stop();
}


void DistributedMatrix::initializePhase(vector3<> k0)
{	if(deriv)
	{	for(std::vector<Cell>& cells: uniqueCells)
			for(Cell& cell: cells)
				cell.phase01 = cis(2*M_PI*dot(cell.iR, k0)) * complex(0., dot(cell.iR, derivRi));
	}
	else
	{	for(std::vector<Cell>& cells: uniqueCells)
			for(Cell& cell: cells)
				cell.phase01 = cis(2*M_PI*dot(cell.iR, k0));
	}
}


void DistributedMatrix::initializePhase(vector3<> k0, std::vector<complex>& phase0)
{	phase0.resize(cellMap.size());
	auto phaseIter = phase0.begin();
	if(deriv)
	{	for(const vector3<int>& iR: cellMap)
			*(phaseIter++) = cis(2*M_PI*dot(iR, k0)) * complex(0., dot(iR, derivRi));
	}
	else
	{	for(const vector3<int>& iR: cellMap)
			*(phaseIter++) = cis(2*M_PI*dot(iR, k0));
	}
}


void DistributedMatrix::collectProc(complex* bufSrc, int ik, int iProc)
{
	//Copy data of current process into location (needed even without MPI if using a different source):
	if(bufSrc and nElems and iProc==mpiUtil->iProcess())
	{	complex* bufDest = (complex*)getResult(ik);
		eblas_copy(bufDest+iElemStart, bufSrc, nElems);
	}
#ifdef MPI_ENABLED
	if(mpiUtil->nProcesses() > 1)
	{	if(iProc == mpiUtil->iProcess())
		{	complex* bufDest = (complex*)getResult(ik);
			std::vector<MPIUtil::Request> requests; requests.reserve(mpiUtil->nProcesses()-1);
			for(int jProc=0; jProc<mpiUtil->nProcesses(); jProc++) if(jProc != iProc)
			{	int nElems_j = iElemStartProc[jProc+1] - iElemStartProc[jProc];
				if(nElems_j)
				{	requests.push_back(MPIUtil::Request());
					mpiUtil->recv(bufDest+iElemStartProc[jProc], nElems_j, jProc, jProc, &(requests.back()));
				}
			}
			if(requests.size()) mpiUtil->waitAll(requests);
		}
		else
		{	if(nElems)
				mpiUtil->send(bufSrc ? bufSrc : buf.data(), nElems, iProc, mpiUtil->iProcess());
		}
		MPI_Barrier(mpiUtil->communicator());
	}
#endif
}
