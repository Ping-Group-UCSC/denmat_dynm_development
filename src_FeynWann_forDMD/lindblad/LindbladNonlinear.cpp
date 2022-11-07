#include <lindblad/Lindblad.h>


LindbladNonlinear::LindbladNonlinear(const LindbladParams& lp) : Lindblad(lp)
{	assert((not lp.spectrumMode) and (not lp.linearized));
}


void LindbladNonlinear::rhoDotScatter()
{	static StopWatch watch("LindbladNonlinear::rhoDotScatter");
	if(not lp.ePhEnabled) return;
	watch.start();

	//Loop over process providing other k data:
	const double prefac = M_PI/nkTot;
	int iProc = mpiWorld->iProcess(); //current process
	for(int jProc=0; jProc<mpiWorld->nProcesses(); jProc++)
	{
		//Make data from jProc available:
		DM1 rho_j(rhoSize[jProc]), rhoDot_j(rhoSize[jProc]);
		if(jProc==iProc)
		{	for(const State& s: state)
				accumRhoHC(0.5*s.rho, rho_j.data()+rhoOffset[s.ik]);
		}
		mpiWorld->bcastData(rho_j, jProc);
		size_t jkStart = kDivision.start(jProc);
		size_t jkStop = kDivision.stop(jProc);

		//Loop over rho1 local to each process:
		for(State& s: state)
		{	const matrix& rho1 = s.rho;
			const matrix rho1bar = bar(rho1);
			matrix& rho1dot = s.rhoDot;

			//Find first entry of GePh whose partner is on jProc (if any):
			std::vector<LindbladFile::GePhEntry>::const_iterator g = std::lower_bound(s.GePh.begin(), s.GePh.end(), jkStart);
			while(g != s.GePh.end())
			{	if(g->jk >= jkStop) break;
				const size_t& ik2 = g->jk;
				const int& nInner2 = nInnerAll[ik2];
				const matrix rho2 = getRho(rho_j.data()+rhoOffset[ik2], nInner2);
				const matrix rho2bar(bar(rho2));
				matrix rho2dot = zeroes(nInner2, nInner2);

				//Loop over all connections to the same partner k:
				while((g != s.GePh.end()) and (g->jk == ik2))
				{
					//Contributions to rho1dot: (+ h.c. added together by accumRhoHC)
					axpyMSMS<false,true>(+prefac, rho1bar, g->Am, rho2, g->Am, rho1dot);
					axpySMSM<false,true>(-prefac, g->Ap, rho2bar, g->Ap, rho1, rho1dot);

					//Contributions to rho2dot: (+ h.c. added together by accumRhoHC)
					axpySMSM<true,false>(+prefac, g->Ap, rho1, g->Ap, rho2bar, rho2dot);
					axpyMSMS<true,false>(-prefac, rho2, g->Am, rho1bar, g->Am, rho2dot);

					//Move to next element:
					g++;
				}
				//Accumulate rho2 gradients:
				accumRhoHC(rho2dot, rhoDot_j.data()+rhoOffset[ik2]);
			}
		}

		//Collect remote contributions:
		mpiWorld->reduceData(rhoDot_j, MPIUtil::ReduceSum, jProc);
		if(jProc==iProc)
		{	for(State& s: state)
				s.rhoDot += 0.5*getRho(rhoDot_j.data()+rhoOffset[s.ik], s.nInner);
		}
	}
	watch.stop();
}
