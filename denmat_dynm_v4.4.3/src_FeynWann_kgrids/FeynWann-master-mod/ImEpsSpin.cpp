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

#include <core/Util.h>
#include <core/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include "FeynWann.h"
#include "Histogram.h"
#include "InputMap.h"
#include "Interp1.h"
#include <core/Units.h>

//Get energy range from an eLoop call:
struct EnergyRange
{	double Emin;
	double Emax;
	
	static void eProcess(const FeynWann::StateE& state, void* params)
	{	EnergyRange& er = *((EnergyRange*)params);
		er.Emin = std::min(er.Emin, state.E.front()); //E is in ascending order
		er.Emax = std::max(er.Emax, state.E.back()); //E is in ascending order
	}
};

//Singularity extrapolation for phonon-assisted:
double extrapCoeff[] = {-19./12, 13./3, -7./4 }; //account for constant, 1/eta and eta^2 dependence
//double extrapCoeff[] = { -1, 2.}; //account for constant and 1/eta dependence
const int nExtrap = sizeof(extrapCoeff)/sizeof(double);

//Collect ImEps contibutions using FeynWann callbacks:
struct CollectImEps
{	const std::vector<double>& dmu;
	double T, invT;
	double GammaS;
	double domega, omegaFull, omegaMax;
	const Interp1* dfInterp; //change in occupations due to perturbation
	size_t nChannels;
	std::vector<Histogram> ImEps;
	Histogram2D ImEps_E;  //ImEpsDelta resolved by carrier density; collected only for first mu
	std::vector<Histogram2D> ImEps_S;
	double prefac;
	double eta; //singularity extrapolation width
	vector3<complex> Ehat;
	double EvMax, EcMin;
	
	
	CollectImEps(const std::vector<double>& dmu, double T, double domega, double omegaFull, double omegaMax, const Interp1* dfInterp)
	: dmu(dmu), T(T), invT(1./T), domega(domega), omegaFull(omegaFull), omegaMax(omegaMax), dfInterp(dfInterp),
		nChannels(dmu.size()*(dfInterp ? 2 : 1)),
		ImEps(nChannels, Histogram(0, domega, omegaFull)),
		ImEps_E(-omegaMax, domega, omegaMax,  0, domega, omegaFull), //ImEpsDelta resolved by carrier density; collected only for first mu
		ImEps_S(3, Histogram2D(-omegaMax, domega, omegaMax,  0, domega, omegaFull)) //ImEpsDelta resolved by spin carrier density; collected only for first mu
	{	logPrintf("Initialized frequency grid: 0 to %lg eV with %d points.\n", ImEps[0].Emax()/eV, ImEps[0].nE);
		EvMax = *std::max_element(dmu.begin(), dmu.end()) + 10*T;
		EcMin = *std::min_element(dmu.begin(), dmu.end()) - 10*T;
	}
	
	void calcStateRelated(const FeynWann::StateE& state, std::vector<diagMatrix>& F)
	{	int nBands = state.E.nRows();
		F.assign(nChannels, diagMatrix(nBands));
		for(unsigned iMu=0; iMu<dmu.size(); iMu++)
			for(int b=0; b<nBands; b++)
				F[iMu][b] = fermi((state.E[b]-dmu[iMu])*invT);
		//Compute perturbed versions if needed:
		diagMatrix dF;
		if(dfInterp)
		{	dF.resize(nBands);
			for(int b=0; b<nBands; b++)
				dF[b] = (*dfInterp)(0,state.E[b]);
			for(unsigned iMu=0; iMu<dmu.size(); iMu++)
				F[iMu+dmu.size()] = F[iMu] + dF;
		}
	}
	
	//---- Direct transitions ----
	void collectDirect(const FeynWann::StateE& state)
	{	int nBands = state.E.nRows();
		//Calculate Fermi fillings and linewidths:
		const diagMatrix& E = state.E;
		std::vector<diagMatrix> F;
		calcStateRelated(state, F);
		//Project dipole matrix elements on field:
		matrix P;
		for(int iDir=0; iDir<3; iDir++)
			P += Ehat[iDir] * state.v[iDir];
		//Collect 
		for(int v=0; v<nBands; v++) if(E[v]<EvMax)
		{	const vector3<>& Sv = state.Svec[v];
			for(int c=0; c<nBands; c++) if(E[c]>EcMin)
			{	double omega = E[c] - E[v]; //energy conservation
				const vector3<>& Sc = state.Svec[c];
				if(omega<domega || omega>=omegaFull) continue; //irrelevant event
				double weight_F = (prefac/(omega*omega)) * P(c,v).norm(); //event weight except for occupation factors
				for(size_t iChannel=0; iChannel<nChannels; iChannel++)
				{	double weight = weight_F * (F[iChannel][v]-F[iChannel][c]);
					ImEps[iChannel].addEvent(omega, weight);
					if(iChannel==0)
					{	ImEps_E.addEvent(E[v], omega, -weight); //hole
						ImEps_E.addEvent(E[c], omega, +weight); //electron
						for (int iDir=0; iDir<3; iDir++)
						{	ImEps_S[iDir].addEvent(E[v], omega, (-weight)*Sv[iDir]); //hole
							ImEps_S[iDir].addEvent(E[c], omega, (+weight)*Sc[iDir]); //electron
						}	
					}
				}
			}
		}
	}
	static void direct(const FeynWann::StateE& state, void* params)
	{	((CollectImEps*)params)->collectDirect(state);
	}
	
	//---- Phonon-assisted transitions ----
	void collectPhonon(const FeynWann::MatrixEph& mat)
	{	int nBands = mat.e1->E.nRows();
		//Calculate Fermi fillings and linewidths:
		const diagMatrix& E1 = mat.e1->E;
		const diagMatrix& E2 = mat.e2->E;
		std::vector<diagMatrix> F1, F2;
		calcStateRelated(*mat.e1, F1);
		calcStateRelated(*mat.e2, F2);
		//Project dipole matrix elements on field:
		matrix P1, P2;
		for(int iDir=0; iDir<3; iDir++)
		{	P1 += Ehat[iDir] * mat.e1->v[iDir];
			P2 += Ehat[iDir] * mat.e2->v[iDir];
		}
		//Bose occupations:
		const diagMatrix& omegaPh = mat.ph->omega;
		int nModes = omegaPh.nRows();
		diagMatrix nPh(nModes);
		for(int iMode=0; iMode<nModes; iMode++)
		{	double omegaPhByT = omegaPh[iMode]/T;
			nPh[iMode] = bose(std::max(1e-3, omegaPhByT)); //avoid 0/0 for zero phonon frequencies
		}
		//Collect
		for(int v=0; v<nBands; v++) if(E1[v]<EvMax)
		{	const vector3<>& Sv = mat.e1->Svec[v];
			for(int c=0; c<nBands; c++) if(E2[c]>EcMin)
			{	const vector3<>& Sc = mat.e2->Svec[c];
				for(int alpha=0; alpha<nModes; alpha++)
				{	for(int ae=-1; ae<=+1; ae+=2) // +/- for phonon absorption or emmision
					{	double omega = E2[c] - E1[v] - ae*omegaPh[alpha]; //energy conservation
						if(omega<domega || omega>=omegaFull) continue; //irrelevant event
						//Effective matrix elements
						std::vector<complex> Meff(nExtrap, 0.);
						for(int i=0; i<nBands; i++) // sum over the intermediate states
						{	complex numA = mat.M[alpha](v,i) * P2(i,c); double denA = E2[i] - (E2[c] - omega);
							complex numB = P1(v,i) * mat.M[alpha](i,c); double denB = E1[i] - (E1[v] + omega);
							double zEta = eta;
							for(int z=0; z<nExtrap; z++)
							{	Meff[z] += ( numA / complex(denA,zEta) + numB / complex(denB,zEta) );
								zEta += eta; //contains (z+1)*eta when evaluating above
							}
						}
						//Singularity extrapolation:
						double MeffSqExtrap = 0.;
						for(int z=0; z<nExtrap; z++)
							MeffSqExtrap += extrapCoeff[z] * Meff[z].norm();
						double weight_F = (prefac/(omega*omega)) * (nPh[alpha] + 0.5*(1.-ae)) * MeffSqExtrap;
						for(size_t iChannel=0; iChannel<nChannels; iChannel++)
						{	double weight = weight_F * (F1[iChannel][v]-F2[iChannel][c]);
							ImEps[iChannel].addEvent(omega, weight);
							if(iChannel==0)
							{	ImEps_E.addEvent(E1[v], omega, -weight); //hole
								ImEps_E.addEvent(E2[c], omega, +weight); //electron	
								for (int iDir=0; iDir<3; iDir++)
								{	ImEps_S[iDir].addEvent(E1[v], omega, (-weight)*Sv[iDir]); //hole
									ImEps_S[iDir].addEvent(E2[c], omega, (+weight)*Sc[iDir]); //electron
								}	
							}
						}
					}
				}
			}
		}
	}
	static void phonon(const FeynWann::MatrixEph& mat, void* params)
	{	((CollectImEps*)params)->collectPhonon(mat);
	}
};

//Lorentzian kernel for an odd function stored on postive frequencies alone:
inline double lorentzianOdd(double omega, double omega0, double breadth)
{	double breadthSq = std::pow(breadth,2);
	return (breadth/M_PI) *
		( 1./(breadthSq + std::pow(omega-omega0, 2))
		- 1./(breadthSq + std::pow(omega+omega0, 2)) );
}
inline void print(FILE* fp, const vector3<complex>& v, const char* format="%lg ")
{	std::fprintf(fp, "[ "); for(int k=0; k<3; k++) fprintf(fp, format, v[k].real()); std::fprintf(fp, "] + 1j*");
	std::fprintf(fp, "[ "); for(int k=0; k<3; k++) fprintf(fp, format, v[k].imag()); std::fprintf(fp, "]\n");
}
inline vector3<complex> normalize(const vector3<complex>& v) { return v * (1./sqrt(v[0].norm() + v[1].norm() + v[2].norm())); }

int main(int argc, char** argv)
{	
	InitParams ip = FeynWann::initialize(argc, argv, "Wannier calculation of imaginary dielectric tensor (ImEps)");

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(ip.inputFilename);
	const int nOffsets = inputMap.get("nOffsets"); assert(nOffsets>0);
	const double omegaMax = inputMap.get("omegaMax") * eV;
	const double T = inputMap.get("T") * Kelvin;
	const double dE = inputMap.get("dE") * eV; //energy resolution used for output and energy conservation
	const vector3<complex> pol = normalize(
		complex(1,0)*inputMap.getVector("polRe", vector3<>(1.,0.,0.)) +  //Real part of polarization
		complex(0,1)*inputMap.getVector("polIm", vector3<>(0.,0.,0.)) ); //Imag part of polarization
	const double GammaS = inputMap.get("GammaS", 0.) * eV; //surface contribution for broadening (default to 0.0 eV)
	const double eta = inputMap.get("eta", 0.1) * eV; //on-shell extrapolation width (default to 0.1 eV)
	const double dmuMin = inputMap.get("dmuMin", 0.) * eV; //optional shift in chemical potential from neutral value; start of range (default to 0)
	const double dmuMax = inputMap.get("dmuMax", 0.) * eV; //optional shift in chemical potential from neutral value; end of range (default to 0)
	const int dmuCount = inputMap.get("dmuCount", 1); assert(dmuCount>0); //number of chemical potential shifts
	const string contribution = inputMap.getString("contribution"); //direct / phonon
	const string runName = inputMap.getString("runName"); //prefix to use for output files

	//Probe mode calculation triggered by non-zero Uabs (will read eDOS.dat and a carrier distribution file):
	const double pumpUabs = inputMap.get("pumpUabs", 0.) * Joule/std::pow(meter,3); //absorbed laser energy per unit volume in Joule/meter^3
	const double pumpOmega = inputMap.get("pumpOmega", 0.) * eV; //pump photon energy in eV (pump pol is what was used in the phase 1 calc)
	const string pumpRunName = pumpUabs ? inputMap.getString("pumpRunName") : string(); //run name to read carrier distribution of pump from
	
	//Check contribution:
	enum ContribType { Direct, Phonon };
	EnumStringMap<ContribType> contribMap(Direct, "Direct", Phonon, "Phonon");
	ContribType contribType;
	if(!contribMap.getEnum(contribution.c_str(), contribType))
		die("Input parameter 'contribution' must be one of %s.\n\n", contribMap.optionList().c_str());
	string fileSuffix = contribMap.getString(contribType);
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nOffsets = %d\n", nOffsets);
	logPrintf("omegaMax = %lg\n", omegaMax);
	logPrintf("T = %lg\n", T);
	logPrintf("dE = %lg\n", dE);
	logPrintf("pol = "); print(globalLog, pol);
	logPrintf("GammaS = %lg\n", GammaS);
	logPrintf("eta = %lg\n", eta);
	logPrintf("dmuMin = %lg\n", dmuMin);
	logPrintf("dmuMax = %lg\n", dmuMax);
	logPrintf("dmuCount = %d\n", dmuCount);
	logPrintf("contribution = %s\n", contribMap.getString(contribType));
	logPrintf("runName = %s\n", runName.c_str());
	if(pumpUabs)
	{	logPrintf("pumpUabs = %lg\n", pumpUabs);
		logPrintf("pumpOmega = %lg\n", pumpOmega);
		logPrintf("pumpRunName = %s\n", pumpRunName.c_str());
	}

	//Initialize FeynWann:
	FeynWannParams fwp;
	fwp.needPhonons = (contribType==Phonon);
	fwp.needVelocity = true;
	fwp.needSpin = true;
	fwp.needLinewidth_ee = false;
	fwp.needLinewidth_ePh = false;
	std::shared_ptr<FeynWann> fw = std::make_shared<FeynWann>(fwp);
	size_t nKeff = nOffsets * (contribType==Direct ? fw->eCountPerOffset() : fw->ePhCountPerOffset());
	logPrintf("Effectively sampled %s: %lu\n", (contribType==Direct ? "nKpts" : "nKpairs"), nKeff);
	
	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		fw = 0;
		FeynWann::finalize();
		return 0;
	}
	logPrintf("\n");
	//Initialize sampling parameters:
	int oStart=0, oStop=0;
	if(mpiGroup->isHead())
		TaskDivision(nOffsets, mpiGroupHead).myRange(oStart, oStop);
	mpiGroup->bcast(oStart);
	mpiGroup->bcast(oStop);
	int noMine = oStop-oStart; //number of offsets handled by current group
	int oInterval = std::max(1, int(round(noMine/50.))); //interval for reporting progress
	
	//Initialize frequency grid:
	const double domega = dE;
	EnergyRange er = { DBL_MAX, -DBL_MAX };
	fw->eLoop(vector3<>(), EnergyRange::eProcess, &er);
	mpiWorld->allReduce(er.Emin, MPIUtil::ReduceMin);
	mpiWorld->allReduce(er.Emax, MPIUtil::ReduceMax);
	double omegaFull = er.Emax - er.Emin;
	
	//dmu array:
	std::vector<double> dmu(dmuCount, dmuMin); //set first value here
	for(int iMu=1; iMu<dmuCount; iMu++) //set remaining values (if any)
		dmu[iMu] = dmuMin + iMu*(dmuMax-dmuMin)/(dmuCount-1);
	
	//Initialize change in fillings if required
	Interp1 dfInterp;
	if(pumpUabs)
	{	//Read carrier distributions and DOS:
		Histogram2D distribDirect("carrierDistrib"+fileSuffix+pumpRunName+".dat", 1./eV, 1./eV, 1.);
		Interp1 dos; dos.init("eDOS.dat", eV, 1.);
		//Calculate and normalize change in fillings:
		diagMatrix dfPert;
		int nE = dos.xGrid.size();
		double dE = dos.dx;
		dfPert.resize(nE);
		double Upert = 0.;
		for(int ie=0; ie<nE; ie++)
		{	const double& Ei = dos.xGrid[ie];
			double dni = distribDirect.interp1(Ei, pumpOmega); //induced carrier number change at given energy
			Upert += dni * Ei * dE; //calculate energy of perturbation
			dfPert[ie] = dni / std::max(dos.yGrid[0][ie], 1e-6); //divide by DOS to get the effective filling change (regularize to avoid Infs)
		}
		dfPert *= pumpUabs / Upert; //normalize to match absorbed laser energy per unit volume
		//Set to dfInterp:
		dfInterp = dos;
		dfInterp.yGrid[0] = dfPert;
	}
	

	CollectImEps cie(dmu, T, domega, omegaFull, omegaMax, (pumpUabs ? &dfInterp : 0));
	cie.prefac = 4. * std::pow(M_PI,2) * fw->spinWeight / (nKeff*fabs(det(fw->R))); //frequency independent part of prefactor
	cie.eta = eta;
	cie.GammaS = GammaS;
	cie.Ehat = pol;
	
	for(int iSpin=0; iSpin<fw->nSpins; iSpin++)
	{	//Update FeynWann for spin channel if necessary:
		if(iSpin>0)
		{	fw = 0; //free memory from previous spin
			fwp.iSpin = iSpin;
			fw = std::make_shared<FeynWann>(fwp);
		}
		logPrintf("\nCollecting ImEps: "); logFlush();
		for(int o=0; o<noMine; o++)
		{	Random::seed(o+oStart); //to make results independent of MPI division
			//Process with a random offset:
			switch(contribType)
			{	case Direct:
				{	vector3<> k0 = fw->randomVector(mpiGroup); //must be constant across group
					fw->eLoop(k0, CollectImEps::direct, &cie);
					break;
				}
				case Phonon:
				{	vector3<> k01 = fw->randomVector(mpiGroup); //must be constant across group
					vector3<> k02 = fw->randomVector(mpiGroup); //must be constant across group
					fw->ePhLoop(k01, k02, CollectImEps::phonon, &cie);
					break;
				}
			}
			//Print progress:
			if((o+1)%oInterval==0) { logPrintf("%d%% ", int(round((o+1)*100./noMine))); logFlush(); }
		}
		logPrintf("done.\n"); logFlush();
	}
	
	//Apply broadening:
	std::vector<Histogram> ImEps(cie.nChannels, Histogram(0, domega, omegaFull));
	Histogram2D ImEps_E(-omegaMax, domega, omegaMax,  0, domega, omegaMax);
	std::vector<Histogram2D> ImEps_S(3, Histogram2D(-omegaMax, domega, omegaMax,  0, domega, omegaMax));
	int iomegaStart, iomegaStop; TaskDivision(ImEps_E.nomega, mpiWorld).myRange(iomegaStart, iomegaStop);
	logPrintf("Applying broadening ... "); logFlush();
	for(size_t iChannel=0; iChannel<cie.nChannels; iChannel++)
	{	for(int iomega=iomegaStart; iomega<iomegaStop; iomega++) //input frequency grid split over MPI
		{	double omegaCur = iomega*domega;
			for(size_t jomega=0; jomega<ImEps[iChannel].out.size(); jomega++) //output frequency grid
			{	double omega = jomega*domega;
				double kernel = lorentzianOdd(omega, omegaCur,0.000367) * domega;
				ImEps[iChannel].out[jomega] += kernel * cie.ImEps[iChannel].out[iomega];
				//Carrier distributions:
				if(iChannel==0 && int(jomega)<ImEps_E.nomega)
				{	const int nE = ImEps_E.nE; assert(nE == cie.ImEps_E.nE);
					for(int iE=0; iE<nE; iE++)
					{	int iOE = iomega*nE + iE;
						int jOE = jomega*nE + iE;
						ImEps_E.out[jOE] += kernel * cie.ImEps_E.out[iOE];
						for (int iDir=0; iDir<3; iDir++)
						{	ImEps_S[iDir].out[jOE] += kernel * cie.ImEps_S[iDir].out[iOE];
						}
					}
				}
			}
		}
		ImEps[iChannel].allReduce(MPIUtil::ReduceSum);
	}
	if(not pumpUabs)
	{	ImEps_E.allReduce(MPIUtil::ReduceSum); ImEps_E.print("carrierDistrib"+fileSuffix+runName +".dat", 1./eV, 1./eV, 1.);
		ImEps_S[0].allReduce(MPIUtil::ReduceSum); ImEps_S[0].print("carrierDistrib"+fileSuffix+runName +"Sx.dat", 1./eV, 1./eV, 1.);
		ImEps_S[1].allReduce(MPIUtil::ReduceSum); ImEps_S[1].print("carrierDistrib"+fileSuffix+runName +"Sy.dat", 1./eV, 1./eV, 1.);
		ImEps_S[2].allReduce(MPIUtil::ReduceSum); ImEps_S[2].print("carrierDistrib"+fileSuffix+runName +"Sz.dat", 1./eV, 1./eV, 1.);
	}
	logPrintf("done.\n"); logFlush();
	
	//Output ImEps:
	if(mpiWorld->isHead())
	{	ofstream ofs("ImEps"+fileSuffix+runName+".dat");
		ofs << "#omega[eV]";
		for(int iMu=0; iMu<dmuCount; iMu++)
			ofs << " ImEps[mu=" << dmu[iMu]/eV << "eV]";
		if(pumpUabs)
		{	for(int iMu=0; iMu<dmuCount; iMu++)
				ofs << " ImEpsPert[mu=" << dmu[iMu]/eV << "eV]";
		}
		ofs << "\n";
		for(size_t iOmega=0; iOmega<ImEps[0].out.size(); iOmega++)
		{	double omega = ImEps[0].Emin + ImEps[0].dE * iOmega;
			ofs << omega/eV;
			for(size_t iChannel=0; iChannel<cie.nChannels; iChannel++)
				ofs << '\t' << ImEps[iChannel].out[iOmega];
			ofs << '\n';
		}
	}
	
	fw = 0;
	FeynWann::finalize();
	return 0;
}
