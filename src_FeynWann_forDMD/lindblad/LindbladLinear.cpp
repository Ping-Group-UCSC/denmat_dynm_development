#include <lindblad/Lindblad.h>


LindbladLinear::LindbladLinear(const LindbladParams& lp)
: LindbladMatrix(lp), nnzD(rhoSizeTot), nnzO(rhoSizeTot)
{
	assert((not lp.spectrumMode) and lp.linearized);

	if(lp.ePhEnabled)
	{
		#ifdef PETSC_ENABLED
		initialize();

		//Initialize sparse time evolution matrix in evolveMat:
		initializeMatrix();
		
		//Finalize matrix assembly:
		CHECKERR(MatAssemblyBegin(evolveMat, MAT_FINAL_ASSEMBLY));
		CHECKERR(MatAssemblyEnd(evolveMat, MAT_FINAL_ASSEMBLY));
		MatInfo info; CHECKERR(MatGetInfo(evolveMat, MAT_GLOBAL_SUM, &info));
		logPrintf("done. Net sparsity: %.0lf non-zero in %lu x %lu matrix (%.1lf%% fill)\n",
			info.nz_used, rhoSizeTot, rhoSizeTot, info.nz_used*100./(rhoSizeTot*rhoSizeTot));
		logFlush();
		
		//Create corresponding vectors:
		CHECKERR(MatCreateVecs(evolveMat, &vRho, &vRhoDot));
		#endif
	}
}


LindbladLinear::~LindbladLinear()
{
	#ifdef PETSC_ENABLED
	cleanup();
	#endif
}


void LindbladLinear::rhoDotScatter()
{	static StopWatch watch("LindbladLinear::rhoDotScatter");
	#ifdef PETSC_ENABLED
	if(not lp.ePhEnabled) return;
	watch.start();

	//Convert drho to PETSc format:
	double* vRhoPtr;
	VecGetArray(vRho, &vRhoPtr);
	eblas_zero(drho.size(), vRhoPtr);
	for(const State& s: state)
		accumRhoHC(0.5*s.drho, vRhoPtr+rhoOffset[s.ik]);
	VecRestoreArray(vRho, &vRhoPtr);

	//Apply sparse operator using PETSc:
	MatMult(evolveMat, vRho, vRhoDot);

	//Extract rhoDot into state from PETSc:
	const double* vRhoDotPtr;
	VecGetArrayRead(vRhoDot, &vRhoDotPtr);
	for(State& s: state)
		s.rhoDot += 0.5 * getRho(vRhoDotPtr+rhoOffset[s.ik], s.nInner); //0.5 to compensate for +HC added later
	VecRestoreArrayRead(vRhoDot, &vRhoDotPtr);
	watch.stop();
	#endif
}


//------ PETSC library initalize and cleanup ------

#ifdef PETSC_ENABLED

void LindbladLinear::initialize()
{	//Create a fake command line for PetscInitialize (condlicts with FW command line):
	int argc = 1;
	char argvBuf[256]; strcpy(argvBuf, "lindblad/run");
	char* argv0 = &argvBuf[0];
	char** argv = &argv0;
	CHECKERR(PetscInitialize(&argc, &argv, (char*)0, ""));
}

void LindbladLinear::cleanup()
{	if(lp.ePhEnabled)
	{	CHECKERR(MatDestroy(&evolveMat));
		CHECKERR(VecDestroy(&vRho));
		CHECKERR(VecDestroy(&vRhoDot));
		CHECKERR(PetscFinalize());
	}
}

#endif
