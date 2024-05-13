#include <lindblad/Lindblad.h>
#include <Histogram.h>
#include <core/Units.h>


//---- implementation of class Lindblad's dynamics functions ----


void Lindblad::calculate()
{
	if(not (lp.pumpEvolve or lp.ePhEnabled or lp.Bext.length_squared()))
	{	//Simple probe - one-shot-pump - probe with no relaxation (or coherent dynamics):
		report(-lp.dt, drho);
		applyPump(); //one-shot optical pump or B-field excitation
		report(0., drho);
	}
	else
	{	//Pump and time evolve with continuous probe:
		//Initialization:
		double tStart = 0.;
		if(readCheckpoint(tStart))
		{	//tStart and stepID already set to end of previously checkpointed run.
			//No other initialization needed.
		}
		else
		{	if(lp.pumpEvolve)
			{	//Set start time to a multiple of dt that covers pulse:
				tStart = -lp.dt * ceil(5.*lp.tau/lp.dt);
				//pump will be included in evolution below
			}
			else
			{	report(-lp.dt, drho); //initial-state report
				applyPump(); //takes care of optical pump or B-field excitation
				tStart = 0.; //integrate will report at t=0 below, before evolving ePh relaxation
			}
		}
		
		//Evolution:
		if(lp.tStep) //Fixed-step integrator:
			integrateFixed(drho, tStart, lp.tStop, lp.tStep, lp.dt);
		else //Adaptive integrator:
			integrateAdaptive(drho, tStart, lp.tStop, lp.tolAdaptive, lp.dt);
	}
}


void Lindblad::applyPump()
{	static StopWatch watch("Lindblad::applyPump"); 
	if(lp.pumpEvolve) return; //only use this function when perturbing instantly
	watch.start();
	//Perturb each k separately:
	for(const State& s: state)
	{	if(lp.pumpBfield)
		{	//Construct Hamiltonian including magnetic field contribution:
			matrix Htot(s.E(s.innerStart, s.innerStart+s.nInner));
			for(int iDir=0; iDir<3; iDir++) //Add Zeeman Hamiltonian
			{	if(spinorial) Htot += (lp.pumpB[iDir] * bohrMagneton * gElectron * 0.5) * s.S[iDir];  //0.5 because |S| in [0, 1]
				if(lp.orbitalZeeman) Htot += (lp.pumpB[iDir] * bohrMagneton) * s.L[iDir];
			}
			//Set rho to Fermi function of this perturbed Hamiltonian:
			diagMatrix Epert; matrix Vpert;
			Htot.diagonalize(Vpert, Epert);
			diagMatrix fPert(s.nInner);
			for(int b=0; b<s.nInner; b++)
				fPert[b] = fermi((Epert[b] - lp.dmu) * lp.invT);
			matrix rhoPert = Vpert * fPert * dagger(Vpert);
			accumRhoHC(0.5*(rhoPert-s.rho0), drho.data()+rhoOffset[s.ik]);
		}
		else
		{	const diagMatrix& rho0 = s.rho0;
			diagMatrix rho0bar = bar(rho0); //1-rho0
			//Compute and apply perturbation:
			matrix P = s.pumpPD; //P-
			matrix Pdag = dagger(P); //P+
			matrix deltaRho;
			for(int s=-1; s<=+1; s+=2)
			{	deltaRho += rho0bar*P*rho0*Pdag - Pdag*rho0bar*P*rho0;
				std::swap(P, Pdag); //P- <--> P+
			}
			accumRhoHC(M_PI * deltaRho, drho.data()+rhoOffset[s.ik]);
		}
	}
	watch.stop();
}


void Lindblad::setState(double t, const DM1& drho, State& s) const
{	s.rhoDot.zero();
	s.drho = getRho(drho.data()+rhoOffset[s.ik], s.nInner);
	s.rho = s.rho0 + s.drho;

	//Calculate and apply phases:
	//--- in diagonal basis
	std::vector<complex> phaseDiag(s.nInner);
	for(int b=0; b<s.nInner; b++)
		phaseDiag[b] = cis(-t * s.E0[b]);
	//--- convert to required basis
	if(s.V0)
	{	s.phase = (s.V0 * phaseDiag) * dagger(s.V0); //construct unitary e^(-i H0 t) transform
		s.drho = s.phase * s.drho * dagger(s.phase); //apply this transform to drho
		s.rho = s.phase * s.rho * dagger(s.phase); //apply this transform to rho
	}
	else
	{	complex* phaseData = s.phase.data();
		complex* drhoData = s.drho.data();
		complex* rhoData = s.rho.data();
		for(int bCol=0; bCol<s.nInner; bCol++)
			for(int bRow=0; bRow<s.nInner; bRow++)
			{	complex phaseCur = phaseDiag[bRow] * phaseDiag[bCol].conj();
				*(phaseData++) = phaseCur; //store multiplicative phase for each element
				*(drhoData++) *= phaseCur; //apply multiplicative phase to drho
				*(rhoData++) *= phaseCur; //apply multiplicative phase to rho
			}
	}
}


void Lindblad::getStateDot(const State& s, DM1& rhoDot) const
{	//Convert rhoDot from Schrodinger to interaction picture:
	matrix rhoDotCur;
	if(s.V0)
		rhoDotCur = dagger(s.phase) * s.rhoDot * s.phase; //reverse unitary transform for phase
	else
	{	rhoDotCur = clone(s.rhoDot);
		complex* rhoDotData = rhoDotCur.data();
		const complex* phaseData = s.phase.data();
		for(int bCol=0; bCol<s.nInner; bCol++)
			for(int bRow=0; bRow<s.nInner; bRow++)
				*(rhoDotData++) *= (phaseData++)->conj(); //conjugated multiplicative phase
	}
	//Set into rhoDot with H.C. term:
	accumRhoHC(rhoDotCur, rhoDot.data()+rhoOffset[s.ik]);
}


DM1 Lindblad::compute(double t, const DM1& drho)
{	double pumpPrefac = 1.0;
	if(lp.pumpEvolve and lp.pumpTau) //Gaussian pulse
		pumpPrefac = (sqrt(M_PI) * exp(-(t*t)/std::pow(lp.pumpTau, 2)) / lp.pumpTau);
	vector3<> Bcur = lp.spinEchoGetB(t);
	
	for(State& s: state)
	{	//Convert interaction picture input to Schrodinger picture within state:
		setState(t, drho, s);
		
		//---- k-diagonal contributions -----
		
		//Pump:
		if(lp.pumpEvolve)
		{	matrix P = s.pumpPD; //P-
			matrix Pdag = dagger(P); //P+
			const matrix rhoBar = bar(s.rho); //1-rho
			for(int sign=-1; sign<=+1; sign+=2)
			{	s.rhoDot += pumpPrefac * (rhoBar * P * s.rho * Pdag
										- Pdag * rhoBar * P * s.rho); //+HC added by getRhoDot()
				std::swap(P, Pdag); //P- <--> P+
			}
		}
		
		//Time-dependent magnetic field contribution:
		if(Bcur.isNonzero())
		{	matrix deltaH;
			for(int iDir=0; iDir<3; iDir++) //Add Zeeman Hamiltonian
			{	if(spinorial) deltaH += (Bcur[iDir] * bohrMagneton * gElectron * 0.5) * s.S[iDir];  //0.5 because |S| in [0, 1]
				if(lp.orbitalZeeman) deltaH += (Bcur[iDir] * bohrMagneton) * s.L[iDir];
			}
			s.rhoDot += complex(0, 1) * s.rho * deltaH; //+HC added by getRhoDot() completes commutator
		}
	}
	
	//Scattering contributions (e-ph, defects) that couple k:
	rhoDotScatter();
	
	//Convert final result back to interaction picture:
	DM1 rhoDot(drho.size());
	for(const State& s: state)
		getStateDot(s, rhoDot);
	
	if(lp.verbose)
	{	//Report current statistics:
		double rhoDotMax = 0., rhoEigMin = +DBL_MAX, rhoEigMax = -DBL_MAX;
		for(const State& s: state)
		{	//max(rhoDot)
			rhoDotMax = std::max(rhoDotMax, s.rhoDot.data()[cblas_izamax(s.rhoDot.nData(), s.rhoDot.data(), 1)].abs());
			//eig(rho):
			matrix V; diagMatrix f;
			s.rho.diagonalize(V, f);
			rhoEigMin = std::min(rhoEigMin, f.front());
			rhoEigMax = std::max(rhoEigMax, f.back());
		}
		mpiWorld->reduce(rhoDotMax, MPIUtil::ReduceMax);
		mpiWorld->reduce(rhoEigMax, MPIUtil::ReduceMax);
		mpiWorld->reduce(rhoEigMin, MPIUtil::ReduceMin);
		logPrintf("\n\tComputed at t[fs]: %lg  max(rhoDot): %lg rhoEigRange: [ %lg %lg ] ",
			t/fs, rhoDotMax, rhoEigMin, rhoEigMax); logFlush();
	}
	else logPrintf("* "); //just a visual progress bar
	logFlush();

	return rhoDot;
}


//Compute x * log(x), handling the limit correctly:
inline double xlogx(double x)
{	return (x <= 0.) ? 0. : x * log(x);
}


void Lindblad::report(double t, const DM1& drho) const
{	static StopWatch watch("Lindblad::report"); watch.start();
	ostringstream ossID; ossID << stepID;

	//Total energy and distributions:
	int nDist = lp.saveDist
		? (spinorial ? 4 : 1) //number distribution only, or also spin distribution
		: 0; //don't save distributions
	std::vector<Histogram> dist(nDist, Histogram(Emin, lp.dE, Emax));
	const double prefac = spinWeight*(1./nkTot); //BZ integration weight
	double Etot = 0., dfMax = 0.; vector3<> Stot;
	double entropy = 0.;
	for(const State& s: state)
	{	setState(t, drho, (State&)s); //update Schrodinger-picture quantities
		
		//Energy and distribution:
		const complex* drhoData = s.drho.data();
		for(int b=0; b<s.nInner; b++)
		{	double weight = prefac * drhoData->real();
			const double& Ecur = s.E[b+s.innerStart];
			Etot += weight * Ecur;
			dfMax = std::max(dfMax, fabs(drhoData->real()));
			if(lp.saveDist)
				dist[0].addEvent(Ecur, weight);
			drhoData += (s.nInner+1); //advance to next diagonal entry
		}
		
		//Entropy:
		diagMatrix rhoEigs; matrix rhoEvecs;
		s.rho.diagonalize(rhoEvecs, rhoEigs);
		for(double r: rhoEigs)
			entropy -= xlogx(r) + xlogx(1. - r);
		
		//Spin distribution (if available):
		if(spinorial)
		{	const complex* drhoData = s.drho.data();
			vector3<const complex*> Sdata; for(int k=0; k<3; k++) Sdata[k] = s.S[k].data();
			std::vector<vector3<>> Sband(s.nInner); //spin expectation by band S_b := sum_a S_ba drho_ab
			for(int b2=0; b2<s.nInner; b2++)
			{	for(int b1=0; b1<s.nInner; b1++)
				{	complex weight = prefac * (*(drhoData++)).conj();
					for(int iDir=0; iDir<3; iDir++)
						Sband[b2][iDir] += (weight * (*(Sdata[iDir]++))).real();
				}
				Stot += Sband[b2];
			}
			//Collect distribution based on per-band spin:
			if(lp.saveDist)
			{	for(int b=0; b<s.nInner; b++)
				{	const double& E = s.E[b+s.innerStart];
					int iEvent; double tEvent;
					if(dist[1].eventPrecalc(E, iEvent, tEvent))
					{	for(int iDir=0; iDir<3; iDir++)
							dist[iDir+1].addEventPrecalc(iEvent, tEvent, Sband[b][iDir]);
					}
				}
			}
		}
	}
	mpiWorld->reduce(Etot, MPIUtil::ReduceSum);
	mpiWorld->reduce(Stot, MPIUtil::ReduceSum);
	mpiWorld->reduce(dfMax, MPIUtil::ReduceMax);
	mpiWorld->reduce(entropy, MPIUtil::ReduceSum);
	for(Histogram& h: dist) h.reduce(MPIUtil::ReduceSum);
	if(mpiWorld->isHead())
	{	//Report step ID and energy:
		logPrintf("Integrate: Step: %4d   t[fs]: %6.1lf   Etot[eV]: %.2le   dfMax: %.2le", stepID, t/fs, Etot/eV, dfMax);
		if(spinorial) logPrintf("   S: [ %16.15lg %16.15lg %16.15lg ]", Stot[0],  Stot[1],  Stot[2]);
		logPrintf("\n"); logFlush();
		
		//Report entropy:
		logPrintf("Entropy: %21.15le\n", entropy);

		//Report rotating frame spin in spin-echo setup:
		if(lp.spinEchoFlipTime)
		{	vector3<> Srot = lp.spinEchoTransform(Stot, t);
			logPrintf("SpinEcho: Srot: [ %16.15lg %16.15lg %16.15lg ]\n", Srot[0],  Srot[1],  Srot[2]);
			logFlush();
		}
		
		//Save distribution functions:
		if(lp.saveDist)
		{	ofstream ofs("dist."+ossID.str());
			ofs << "#E-mu/VBM[eV] n[eV^-1]";
			if(spinorial)
				ofs << "Sx[eV^-1] Sy[eV^-1] Sz[eV^-1]";
			ofs << "\n";
			for(int iE=0; iE<dist[0].nE; iE++)
			{	double E = Emin + iE*lp.dE;
				ofs << E/eV;
				for(int iDist=0; iDist<nDist; iDist++)
					ofs << '\t' << dist[iDist].out[iE]*eV;
				ofs << '\n';
			}
		}
	}
	
	//Other file outputs:
	writeCheckpoint(t);
	writeImEps("imEps." + ossID.str());
	((Lindblad*)this)->stepID++; //Increment stepID
	watch.stop();
}
