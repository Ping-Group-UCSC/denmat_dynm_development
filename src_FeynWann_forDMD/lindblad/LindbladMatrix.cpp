#include <lindblad/Lindblad.h>


void LindbladMatrix::initializeMatrix()
{
	assert(lp.ePhEnabled);

	//Select target based on spectrum mode:
	LindbladLinear* ll = NULL;
	LindbladSpectrum* ls = NULL;
	int nPasses = 0;
	if(lp.spectrumMode)
	{	ls = (LindbladSpectrum*)this;
		nPasses = 1;
		logPrintf("Initializing time evolution operator ... ");
	}
	else
	{	ll = (LindbladLinear*)this;
		nPasses = 2;  //first pass for size determination, second pass to set the entries
		logPrintf("Determining sparsity structure of time evolution operator ... ");
	}
	logFlush();
	
	//Loop over matrix elements
	for(int iPass=0; iPass<nPasses; iPass++)
	{	State* sPtr = state.data();
		for(size_t ik1=ikStart; ik1<ikStop; ik1++)
		{	State& s = *(sPtr++);
			const double* E1 = &(s.E[s.innerStart]);
			const int& nRhoPrev1 = nRhoPrev[ik1];
			const int& nInner1 = nInnerAll[ik1];
			const int N1 = nInner1*nInner1; //number of density matrix entries
			const int whose1 = mpiWorld->iProcess();
			const diagMatrix& f1 = s.rho0;
			const diagMatrix f1bar = bar(f1);
			#ifdef SCALAPACK_ENABLED
			//Coherent evolution (only in spectrum mode):
			if(lp.spectrumMode)
			{	for(int a=0; a<nInner1; a++)
					for(int b=0; b<nInner1; b++)
						if(a != b)
						{	int iRow = nRhoPrev1+a+b*nInner1;
							int iCol = nRhoPrev1+b+a*nInner1;
							double Ediff = E1[b]-E1[a];
							int localIndex, whose = ls->bcm->globalIndex(iRow, iCol, localIndex);
							ls->evolveEntries[whose].push_back(std::make_pair(Ediff,localIndex));
						}
			}
			#endif
			//Electron-phonon part:
			const double prefacEph = 2*M_PI/nkTot; //factor of 2 from the +h.c. contribution
			std::vector<LindbladFile::GePhEntry>::iterator g = s.GePh.begin();
			while(g != s.GePh.end())
			{	const size_t& ik2 = g->jk;
				const int& nInner2 = nInnerAll[ik2];
				const int N2 = nInner2*nInner2; //number of density matrix entries
				const int& nRhoPrev2 = nRhoPrev[ik2];
				const int whose2 = whose(ik2);
				const double* E2 = &(Eall[nInnerPrev[ik2]]);
				diagMatrix f2(nInner2);
				for(int b2=0; b2<nInner2; b2++)
					f2[b2] = fermi(lp.invT * (E2[b2] - lp.dmu));
				const diagMatrix f2bar = bar(f2);
				//Skip combinations if necessary based on valleyMode:
				bool shouldSkip = false; //must skip after iterating over all matching k2 below (else endless loop!)
				if((lp.valleyMode==ValleyIntra) and (isKall[ik1]!=isKall[ik2])) shouldSkip=true; //skip intervalley scattering
				if((lp.valleyMode==ValleyInter) and (isKall[ik1]==isKall[ik2])) shouldSkip=true; //skip intravalley scattering
				//Store results in dense complex blocks of the superoperator first:
				matrix L12 = zeroes(N1,N2); complex* L12data = L12.data();
				matrix L21 = zeroes(N2,N1); complex* L21data = L21.data();
				matrix L11 = zeroes(N1,N1); complex* L11data = L11.data();
				matrix L22 = zeroes(N2,N2); complex* L22data = L22.data();
				#define L(i,a,b, j,c,d) L##i##j##data[L##i##j.index(a+b*nInner##i, c+d*nInner##j)] //access superoperator block element
				//Loop over all connections to the same ik2:
				while((g != s.GePh.end()) and (g->jk == ik2))
				{	g->G.init(nInner1, nInner2);
					g->initA(lp.T, lp.defectFraction);
					//Loop over A- and A+
					for(int pm=0; pm<2; pm++) 
					{	const SparseMatrix& Acur = pm ? g->Ap : g->Am;
						const diagMatrix& f1cur = pm ? f1 : f1bar;
						const diagMatrix& f2cur = pm ? f2bar : f2;
						//Loop oover all pairs of non-zero entries:
						for(const SparseEntry& s1: Acur)
						{	int a = s1.i, b = s1.j; //to match derivation's notation
							for(const SparseEntry& s2: Acur)
							{	int c = s2.i, d = s2.j; //to match derivation's notation
								complex M = prefacEph * (s1.val * s2.val.conj());
								L(1,a,c, 2,b,d) += f1cur[a] * M;
								L(2,d,b, 1,c,a) += f2cur[d] * M;
								if(b == d) for(int e=0; e<nInner1; e++) L(1,e,c, 1,e,a) -= f2cur[b] * M;
								if(a == c) for(int e=0; e<nInner2; e++) L(2,e,b, 2,e,d) -= f1cur[c] * M;
							}
						}
					}
					//Move to next element:
					g++;
				}
				if(shouldSkip) continue;
				#undef L
				//Convert from complex to real input and real outputs (based on h.c. symmetry):
				#define CreateRandInv(i) \
					SparseMatrix R##i(N##i,N##i,2*N##i), Rinv##i(N##i,N##i,2*N##i); \
					for(int a=0; a<nInner##i; a++) \
					{	for(int b=0; b<a; b++) \
						{	int ab = a+b*nInner##i, ba = b+a*nInner##i; \
							R##i.push_back(SparseEntry{ab,ab,complex(1,0)}); R##i.push_back(SparseEntry{ab,ba,complex(0,+1)}); \
							R##i.push_back(SparseEntry{ba,ab,complex(1,0)}); R##i.push_back(SparseEntry{ba,ba,complex(0,-1)}); \
							Rinv##i.push_back(SparseEntry{ab,ab,complex(+0.5,0)}); Rinv##i.push_back(SparseEntry{ab,ba,complex(+0.5,0)}); \
							Rinv##i.push_back(SparseEntry{ba,ab,complex(0,-0.5)}); Rinv##i.push_back(SparseEntry{ba,ba,complex(0,+0.5)}); \
						} \
						int aa = a+a*nInner##i; \
						R##i.push_back(SparseEntry{aa,aa,1.}); \
						Rinv##i.push_back(SparseEntry{aa,aa,1.}); \
					}
				CreateRandInv(1)
				CreateRandInv(2)
				#undef CreateRandInv
				L12 = Rinv1 * (L12 * R2);
				L21 = Rinv2 * (L21 * R1);
				L11 = Rinv1 * (L11 * R1);
				L22 = Rinv2 * (L22 * R2);
				//Extract / count / set non-zero entries depending on the mode and pass:
				if(lp.spectrumMode)
				{
					#ifdef SCALAPACK_ENABLED
					#define EXTRACT_NNZ(i,j) \
					{	const complex* data = L##i##j.data(); \
						for(int col=0; col<L##i##j.nCols(); col++) \
						{	for(int row=0; row<L##i##j.nRows(); row++) \
							{	double M = (data++)->real(); \
								if(M) \
								{	int iRow = row+nRhoPrev##i; \
									int iCol = col+nRhoPrev##j; \
									int localIndex, whose = ls->bcm->globalIndex(iRow, iCol, localIndex); \
									ls->evolveEntries[whose].push_back(std::make_pair(M,localIndex)); \
								} \
							} \
						} \
					}
					EXTRACT_NNZ(1,2)
					EXTRACT_NNZ(2,1)
					EXTRACT_NNZ(1,1)
					EXTRACT_NNZ(2,2)
					#undef EXTRACT_NNZ
					#endif
				}
				else
				{
					#ifdef PETSC_ENABLED
					if(iPass == 0)
					{	//Count non-zero matrix elements:
						#define COUNT_NNZ(i,j) \
						{	std::vector<int>& nnz = (whose##i == whose##j) ? ll->nnzD : ll->nnzO; \
							const complex* data = L##i##j.data(); \
							for(int col=0; col<L##i##j.nCols(); col++) \
							{	for(int row=0; row<L##i##j.nRows(); row++) \
								{	double M = (data++)->real(); \
									if(M) nnz[row+nRhoPrev##i]++; \
								} \
							} \
						}
						COUNT_NNZ(1,2)
						COUNT_NNZ(2,1)
						COUNT_NNZ(1,1)
						COUNT_NNZ(2,2)
						#undef COUNT_NNZ
					}
					else
					{	//Set matrix elements in PETSc matrix:
						#define SET_NNZ(i,j) \
						{	const complex* data = L##i##j.data(); \
							for(int col=0; col<L##i##j.nCols(); col++) \
							{	for(int row=0; row<L##i##j.nRows(); row++) \
								{	double M = (data++)->real(); \
									if(M) CHECKERR(MatSetValue(ll->evolveMat, row+nRhoPrev##i, col+nRhoPrev##j, M, ADD_VALUES)); \
								} \
							} \
						}
						SET_NNZ(1,2)
						SET_NNZ(2,1)
						SET_NNZ(1,1)
						SET_NNZ(2,2)
						#undef SET_NNZ
					}
					#endif
				}
			}
			if(iPass+1 == nPasses) s.GePh.clear(); //no longer needed; optimize memory
		}
		if(lp.spectrumMode) logPrintf("done.\n"); 
		#ifdef PETSC_ENABLED
		else
		{	if(iPass == 0)
			{	//Allocate matrix knowing sparsity structure at the end of the first pass
				int N = rhoSizeTot;
				int Nmine = rhoSize[mpiWorld->iProcess()];
				MPI_Allreduce(MPI_IN_PLACE, ll->nnzD.data(), N, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
				MPI_Allreduce(MPI_IN_PLACE, ll->nnzO.data(), N, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
				for(size_t i=rhoOffsetGlobal; i<rhoOffsetGlobal+Nmine; i++)
				{	ll->nnzD[i] = std::min(ll->nnzD[i], Nmine);
					ll->nnzO[i] = std::min(ll->nnzO[i], N - Nmine);
				}
				logPrintf("done.\nInitializing PETSc sparse matrix for time evolution ... "); logFlush();
				CHECKERR(MatCreateAIJ(PETSC_COMM_WORLD, Nmine, Nmine, N, N,
					0, ll->nnzD.data()+rhoOffsetGlobal, 0, ll->nnzO.data()+rhoOffsetGlobal, &ll->evolveMat));
				CHECKERR(MatSetOption(ll->evolveMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
			}
			//Finalize matrix assembly in calling routine after pass 2
		}
		#endif
	}
}
