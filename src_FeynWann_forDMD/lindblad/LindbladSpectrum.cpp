#include <lindblad/Lindblad.h>
#include <core/Units.h>


LindbladSpectrum::LindbladSpectrum(const LindbladParams& lp)
: LindbladMatrix(lp), evolveEntries(mpiWorld->nProcesses())
{
	assert(lp.spectrumMode);
	assert(lp.ePhEnabled);

	#ifdef SCALAPACK_ENABLED
	bcm = std::make_shared<BlockCyclicMatrix>(rhoSizeTot, lp.blockSize, mpiWorld); //ScaLAPACK wrapper object

	//Initialize non-zero matrix elements in evolveEntries:
	initializeMatrix();
	
	//Convert to dense matrix for ScaLAPACK:
	logPrintf("Converting to block-cyclic distributed dense matrix ... "); logFlush();
	//--- sync sizes of remote pieces:
	std::vector<size_t> nEntriesFromProc(mpiWorld->nProcesses());
	{	std::vector<size_t> nEntriesToProc(mpiWorld->nProcesses());
		std::vector<MPIUtil::Request> requests(2*(mpiWorld->nProcesses()-1));
		int iRequest = 0;
		for(int jProc=0; jProc<mpiWorld->nProcesses(); jProc++)
			if(jProc != mpiWorld->iProcess())
			{	nEntriesToProc[jProc] = evolveEntries[jProc].size();
				mpiWorld->send(&nEntriesToProc[jProc], 1, jProc, 0, &requests[iRequest++]);
				mpiWorld->recv(&nEntriesFromProc[jProc], 1, jProc, 0, &requests[iRequest++]);
				
			}
		mpiWorld->waitAll(requests);
	}
	//--- transfer remote pieces:
	std::vector<std::vector<std::pair<double,int>>> evolveEntriesMine(mpiWorld->nProcesses());
	{	std::vector<MPIUtil::Request> requests(2*(mpiWorld->nProcesses()-1));
		int iRequest = 0;
		for(int jProc=0; jProc<mpiWorld->nProcesses(); jProc++)
			if(jProc == mpiWorld->iProcess())
				std::swap(evolveEntries[jProc], evolveEntriesMine[jProc]);
			else
			{	evolveEntriesMine[jProc].resize(nEntriesFromProc[jProc]);
				MPI_Irecv(evolveEntriesMine[jProc].data(), evolveEntriesMine[jProc].size(), MPI_DOUBLE_INT, jProc, 1, MPI_COMM_WORLD, &requests[iRequest++]);
				MPI_Isend(evolveEntries[jProc].data(), evolveEntries[jProc].size(), MPI_DOUBLE_INT, jProc, 1, MPI_COMM_WORLD, &requests[iRequest++]);
			}
		mpiWorld->waitAll(requests);
		evolveEntries.clear();
	}
	//--- set to dense matrix:
	size_t nNZ = 0;
	evolveMat.assign(bcm->nDataMine, 0.);
	for(const auto& entries: evolveEntriesMine)
	{	for(const std::pair<double,int>& entry: entries)
			evolveMat[entry.second] += entry.first;
		nNZ += entries.size();
	}
	mpiWorld->allReduce(nNZ, MPIUtil::ReduceSum);
	logPrintf("done. Total terms: %lu in %lu x %lu matrix (%.1lf%% fill)\n",
		nNZ, rhoSizeTot, rhoSizeTot, nNZ*100./(rhoSizeTot*rhoSizeTot));
	logFlush();
	
	//Compute spin and magnetic field vectors:
	double Bmag = lp.pumpB.length(); //perturbation strength set by input, but all components calculated
	if(not Bmag)
	{	Bmag = 1.*Tesla;
		logPrintf("Setting test |B| = 1 Tesla for B-field perturbation matrix. (Use pumpB to override if needed.)\n");
	}
	logPrintf("Initializing spin matrix elements ... "); logFlush();
	size_t rhoSizeMine = drho.size();
	DM1 spinMatPert(rhoSizeMine*6); //6 columns: Sx Sy Sz dRho(Bx) dRho(By) dRho(Bz)
	const State* sPtr = state.data();
	const double prefac = spinWeight*(1./nkTot); //BZ integration weight
	for(size_t ik=ikStart; ik<ikStop; ik++)
	{	const State& s = *(sPtr++);
		for(int iDir=0; iDir<3; iDir++)
		{	//Spin matrix element to column iDir:
			matrix OSmat = s.S[iDir] - 0.5*diag(s.S[iDir]); //S with an overlap factor such that S^rho = Tr[rho*S]
			accumRhoHC(prefac*OSmat, spinMatPert.data()+(rhoOffset[ik]+iDir*rhoSizeMine));
			//Magnetic field perturbation to column 3+iDir:
			matrix Htot(s.E(s.innerStart, s.innerStart+s.nInner));
			Htot -= Bmag * s.S[iDir];
			//--- compute Fermi function perturbation:
			diagMatrix Epert; matrix Vpert;
			Htot.diagonalize(Vpert, Epert);
			diagMatrix fPert(s.nInner);
			for(int b=0; b<s.nInner; b++)
				fPert[b] = fermi((Epert[b] - lp.dmu) * lp.invT);
			matrix rhoPert = Vpert * fPert * dagger(Vpert);
			accumRhoHC(0.5*(rhoPert-s.rho0), spinMatPert.data()+(rhoOffset[ik]+(iDir+3)*rhoSizeMine));
		}
	}
	
	//Redistribute to match ScaLAPACK matrices:
	spinMat.resize(bcm->nRowsMine*3); //spin matrix
	spinPert.resize(bcm->nRowsMine*3); //B-field perturbation
	int jProc = mpiWorld->iProcess();
	int iProcPrev = positiveRemainder(mpiWorld->iProcess()-1, mpiWorld->nProcesses());
	int iProcNext = positiveRemainder(mpiWorld->iProcess()+1, mpiWorld->nProcesses());
	for(int iProcShift=0; iProcShift<mpiWorld->nProcesses(); iProcShift++)
	{	//Set local matrix elements from spinMat to spinMatDense:
		int iRowStart = nRhoPrev[kDivision.start(jProc)]; //global start row of current data block
		int iRowStop = nRhoPrev[kDivision.stop(jProc)]; //global stop row of current data block
		int nRowsCur = rhoSize[jProc];
		assert(iRowStop - iRowStart == nRowsCur);
		int iRowMineStart, iRowMineStop; //local row indices that match
		bcm->getRange(bcm->iRowsMine, iRowStart, iRowStop, iRowMineStart, iRowMineStop);
		for(int iRowMine=iRowMineStart; iRowMine<iRowMineStop; iRowMine++)
		{	int iRow = bcm->iRowsMine[iRowMine];
			for(int iCol=0; iCol<3; iCol++)
			{	spinMat[iRowMine+iCol*bcm->nRowsMine] = spinMatPert[(iRow-iRowStart)+iCol*nRowsCur];
				spinPert[iRowMine+iCol*bcm->nRowsMine] = spinMatPert[(iRow-iRowStart)+(iCol+3)*nRowsCur];
			}
		}
		//Circulate spinMat in communication ring:
		if((iProcShift+1) == mpiWorld->nProcesses()) break;
		int jProcNext = (jProc + 1) % mpiWorld->nProcesses();
		DM1 spinMatPertNext(rhoSize[jProcNext]*6);
		std::vector<MPIUtil::Request> request(2);
		mpiWorld->sendData(spinMatPert, iProcPrev, iProcShift, &request[0]);
		mpiWorld->recvData(spinMatPertNext, iProcNext, iProcShift, &request[1]);
		mpiWorld->waitAll(request);
		std::swap(spinMatPert, spinMatPertNext);
		jProc = jProcNext;
	}
	logPrintf("done.\n");
	#endif
}


void LindbladSpectrum::calculate()
{
	#ifdef SCALAPACK_ENABLED
	//Diagonalize:
	BlockCyclicMatrix::Buffer VL, VR;
	std::vector<complex> evals = bcm->diagonalize(evolveMat, VR, VL, lp.diagMethod, false); //diagonalize
	bcm->checkDiagonalization(evolveMat, VR, VL, evals); //check diagonalization
	
	//Transform other quantities to eigenbasis:
	BlockCyclicMatrix::Buffer spinPert, spinMat; //transformed versions of spinPert and spinMat in class
	bcm->matMultVec(1., VL, this->spinPert, spinPert); //weight of each eigenmode in each B-field perturbation
	bcm->matMultVec(1., VR, this->spinMat, spinMat); //spin matrix elements of each eigenmode
	
	//Report eigenvalues:
	logPrintf("\n%19s %19s %19s %19s %19s %19s %19s %19s\n",
		"Re(eig)", "Im(eig)", "rho1(Bx)", "rho1(By)", "rho1(Bz)", "Sx", "Sy", "Sz");
	int nBlocks = ceildiv(bcm->N, lp.blockSize);
	for(int iBlock=0; iBlock<nBlocks; iBlock++)
	{	//Make block of eigenvector overlaps on all processes:
		int whose = iBlock % bcm->nProcsCol;
		int iEigStart = iBlock * lp.blockSize;
		int iEigStop = std::min((iBlock+1) * lp.blockSize, bcm->N);
		int blockSizeCur = iEigStop-iEigStart;
		BlockCyclicMatrix::Buffer spinMatBlock(blockSizeCur*3), spinPertBlock(blockSizeCur*3);
		if(whose == bcm->iProcCol)
		{	int eigStartLocal = (iBlock / bcm->nProcsCol) * lp.blockSize;
			for(int j=0; j<3; j++)
				for(int i=0; i<blockSizeCur; i++)
				{	spinPertBlock[i+j*blockSizeCur] = spinPert[eigStartLocal+i+j*bcm->nColsMine];
					spinMatBlock[i+j*blockSizeCur] = spinMat[eigStartLocal+i+j*bcm->nColsMine];
				}
		}
		bcm->mpiRow->bcastData(spinPertBlock, whose);
		bcm->mpiRow->bcastData(spinMatBlock, whose);
		for(int i=0; i<blockSizeCur; i++)
		{	int iEig = iEigStart+i;
			logPrintf("%19.12le %19.12le %19.12le %19.12le %19.12le %19.12le %19.12le %19.12le\n",
				evals[iEig].real(), evals[iEig].imag(),
				spinPertBlock[i], spinPertBlock[i+blockSizeCur], spinPertBlock[i+2*blockSizeCur],
				spinMatBlock[i], spinMatBlock[i+blockSizeCur], spinMatBlock[i+2*blockSizeCur] );
			logFlush();
		}
	}
	logPrintf("\n");

	//Eigenvector output:
	if(lp.evecFile != "None")
	{	logPrintf("Writing eigenvectors (VR) to '%s' ... ", lp.evecFile.c_str()); logFlush();
		bcm->writeMatrix(VR, lp.evecFile.c_str());
		logPrintf("done.\n");
	}
	#endif
}
