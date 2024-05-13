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
#include <core/Units.h>

//Get energy and speed range from an eLoop call:
struct EnergyRange
{	double Emin;
	double Emax;
	double vMax;
	
	static void eProcess(const FeynWann::StateE& state, void* params)
	{	EnergyRange& er = *((EnergyRange*)params);
		er.Emin = std::min(er.Emin, state.E.front()); //E is in ascending order
		er.Emax = std::max(er.Emax, state.E.back()); //E is in ascending order
		for(const vector3<>& v: state.vVec)
			er.vMax = std::max(er.vMax, v.length()); //v is not ordered
	}
};

//Singularity extrapolation for phonon-assisted:
double extrapCoeff[] = {-19./12, 13./3, -7./4 }; //account for constant, 1/eta and eta^2 dependence
//double extrapCoeff[] = { -1, 2.}; //account for constant and 1/eta dependence
const int nExtrap = sizeof(extrapCoeff)/sizeof(double);

const double c = 137.035999084; //speed of light in atomic units = 1/(fine structure constant)

//Collect ImEps contibutions using FeynWann callbacks:
struct CollectImEps
{	const std::vector<double>& dmu;
	double T, invT;
	double GammaS;
	double domega, omegaFull, omegaMax;
	bool needLW;
	double dv, vMax;
	std::vector<Histogram> ImEps, breadth, breadthDen;
	Histogram2D ImEps_E; //ImEpsDelta resolved by carrier energy; collected only for first mu
	Histogram2D ImEps_v; //ImEpsDelta resolved by carrier speed; collected only for first mu
	double prefac;
	double eta; //singularity extrapolation width
	vector3<complex> Ehat;
	double EvMax, EcMin;
	
	//Optional emission collection (only for first dmu):
	const bool emission;
	const double emissionPrefac; //prefactor to omega^3 in ratio of emission to ImEps phase-space prefactors
	const double selfEmissionPrefac; //prefactor to omega in ratio of self-emission to ImEps phase-space prefactors
	std::shared_ptr<Histogram2D> Emission_E; //carrier and frequency-resolved emission rate
	std::shared_ptr<Histogram2D> selfEmission_E; //carrier and frequency-resolved self-emission rate (from unscattered carriers)
	
	CollectImEps(const std::vector<double>& dmu, double T, double domega, double omegaFull, double omegaMax, bool needLW, double dv, double vMax, bool emission)
	: dmu(dmu), T(T), invT(1./T), domega(domega), omegaFull(omegaFull), omegaMax(omegaMax), needLW(needLW), dv(dv), vMax(vMax),
		ImEps(dmu.size(), Histogram(0, domega, omegaFull)),
		breadth(dmu.size(), Histogram(0, domega, omegaFull)),
		breadthDen(dmu.size(), Histogram(0, domega, omegaFull)),
		ImEps_E(-omegaMax, domega, omegaMax,  0, domega, omegaFull),
		ImEps_v(-vMax, dv, vMax,  0, domega, omegaFull),
		emission(emission),
		emissionPrefac(1.0 / (std::pow(M_PI, 2) * std::pow(c, 3))),
		selfEmissionPrefac(4.0 / std::pow(c, 3))
	{	logPrintf("Initialized frequency grid: 0 to %lg eV with %d points.\n", ImEps[0].Emax()/eV, ImEps[0].nE);
		EvMax = *std::max_element(dmu.begin(), dmu.end()) + 10*T;
		EcMin = *std::min_element(dmu.begin(), dmu.end()) - 10*T;
		if(emission)
		{	Emission_E = std::make_shared<Histogram2D>(-omegaMax, domega, omegaMax,  0, domega, omegaMax);
			selfEmission_E = std::make_shared<Histogram2D>(-omegaMax, domega, omegaMax,  0, domega, omegaMax);
		}
	}
	
	void calcStateRelated(const FeynWann::StateE& state, std::vector<diagMatrix>& F, std::vector<diagMatrix>& ImE)
	{	int nBands = state.E.nRows();
		F.assign(dmu.size(), diagMatrix(nBands));
		if(needLW) ImE.assign(dmu.size(), state.ImSigma_ee); //e-e part
		for(unsigned iMu=0; iMu<dmu.size(); iMu++)
		{	for(int b=0; b<nBands; b++)
			{	F[iMu][b] = fermi((state.E[b]-dmu[iMu])*invT);
				if(needLW) ImE[iMu][b] += state.ImSigma_ePh(b, F[iMu][b]);
			}
		}
	}
	
	//---- Direct transitions ----
	void collectDirect(const FeynWann::StateE& state)
	{	int nBands = state.E.nRows();
		//Calculate Fermi fillings and linewidths:
		const diagMatrix& E = state.E;
		std::vector<diagMatrix> F, ImE;
		calcStateRelated(state, F, ImE);
		//Project momentum matrix elements on field:
		matrix P;
		for(int iDir=0; iDir<3; iDir++)
			P += Ehat[iDir] * state.v[iDir];
		//Collect 
		for(int v=0; v<nBands; v++)
		{	for(int c=0; c<nBands; c++)
			{	double omega = E[c] - E[v]; //energy conservation
				if((omega < domega) or (omega >= omegaFull)) continue; //always irrelevant
				bool relevantImEps = (E[v] < EvMax) and (E[c] > EcMin);
				bool relevantEmission = emission and ((E[c] < EvMax) or (E[v] > EcMin)) and (omega <= omegaMax);
				if(not (relevantImEps or relevantEmission)) continue;
				double weight_F = (prefac/(omega*omega)) * P(c,v).norm(); //event weight except for occupation factors
				if(relevantImEps)
					for(unsigned iMu=0; iMu<dmu.size(); iMu++)
					{	double weight = weight_F * (F[iMu][v]-F[iMu][c]);
						ImEps[iMu].addEvent(omega, weight);
						if(needLW) breadth[iMu].addEvent(omega, weight*(ImE[iMu][c]+ImE[iMu][v]+GammaS));
						if(iMu==0)
						{	ImEps_E.addEvent(E[v], omega, -weight); //hole
							ImEps_E.addEvent(E[c], omega, +weight); //electron
							//speed distribution:
							ImEps_v.addEvent(-state.vVec[v].length(), omega, -weight); //hole
							ImEps_v.addEvent(+state.vVec[c].length(), omega, +weight); //electron
						}
					}
				if(relevantEmission)
				{	double weight = weight_F * emissionPrefac * std::pow(omega, 3);
					Emission_E->addEvent(E[v], omega, -weight * F[0][c]); //hole
					Emission_E->addEvent(E[c], omega, +weight * (1.-F[0][v])); //electron
				}
				if(relevantEmission and relevantImEps)
				{	double weight = weight_F * (F[0][v]-F[0][c]) //ImEps part
						* selfEmissionPrefac * omega * P(c,v).norm(); //emission part
					selfEmission_E->addEvent(E[v], omega, weight * F[0][c]); //hole
					selfEmission_E->addEvent(E[c], omega, weight * (1.-F[0][v])); //electron
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
		std::vector<diagMatrix> F1, F2, ImE1, ImE2;
		calcStateRelated(*mat.e1, F1, ImE1);
		calcStateRelated(*mat.e2, F2, ImE2);
		//Project momentum matrix elements on field:
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
		for(int v=0; v<nBands; v++)
		{	for(int c=0; c<nBands; c++)
			{	for(int alpha=0; alpha<nModes; alpha++)
				{	for(int ae=-1; ae<=+1; ae+=2) // +/- for phonon absorption or emmision
					{	double omega = E2[c] - E1[v] - ae*omegaPh[alpha]; //energy conservation
						if((omega < domega) or (omega >= omegaFull)) continue; //always irrelevant
						bool relevantImEps = (E1[v] < EvMax) and (E2[c] > EcMin);
						bool relevantEmission = emission and ((E2[c] < EvMax) or (E1[v] > EcMin)) and (omega <= omegaMax);
						if(not (relevantImEps or relevantEmission)) continue;

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
						if(relevantImEps)
							for(unsigned iMu=0; iMu<dmu.size(); iMu++)
							{	double weight = weight_F * (F1[iMu][v]-F2[iMu][c]);
								ImEps[iMu].addEvent(omega, weight);
								if(needLW)
								{	breadth[iMu].addEvent(omega, fabs(weight)*(ImE2[iMu][c]+ImE1[iMu][v]+GammaS));
									breadthDen[iMu].addEvent(omega, fabs(weight));
								}
								if(iMu==0)
								{	ImEps_E.addEvent(E1[v], omega, -weight); //hole
									ImEps_E.addEvent(E2[c], omega, +weight); //electron
									//speed distribution:
									ImEps_v.addEvent(-mat.e1->vVec[v].length(), omega, -weight); //hole
									ImEps_v.addEvent(+mat.e2->vVec[c].length(), omega, +weight); //electron
								}
							}
						if(relevantEmission)
						{	double weight = weight_F * emissionPrefac * std::pow(omega, 3);
							Emission_E->addEvent(E1[v], omega, -weight * F2[0][c]); //hole
							Emission_E->addEvent(E2[c], omega, +weight * (1.-F1[0][v])); //electron
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
//Gaussian kernel for an odd function stored on postive frequencies alone:
inline double gaussianOdd(double omega, double omega0, double sigma)
{	double sigmaInvSq = 1./(sigma*sigma);
	return 1./(sqrt(2*M_PI)*sigma) *
		( exp(-0.5*sigmaInvSq*std::pow(omega-omega0, 2))
		- exp(-0.5*sigmaInvSq*std::pow(omega+omega0, 2)) );
}

int main(int argc, char** argv)
{	
	InitParams ip = FeynWann::initialize(argc, argv, "Wannier calculation of imaginary dielectric tensor (ImEps)");

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(ip.inputFilename);
	const int nOffsets = inputMap.get("nOffsets"); assert(nOffsets>0);
	const double omegaMax = inputMap.get("omegaMax") * eV;
	const double T = inputMap.get("T") * Kelvin;
	const double dE = inputMap.get("dE") * eV; //energy resolution used for output and energy conservation
	const vector3<> polRe = inputMap.getVector("polRe", vector3<>(1.,0.,0.)); //Real part of polarization
	const vector3<> polIm = inputMap.getVector("polIm", vector3<>(0.,0.,0.)); //Imag part of polarization
	const double GammaS = inputMap.get("GammaS", 0.) * eV; //surface contribution for broadening (default to 0.0 eV)
	const double eta = inputMap.get("eta", 0.1) * eV; //on-shell extrapolation width (default to 0.1 eV)
	const double dmuMin = inputMap.get("dmuMin", 0.) * eV; //optional shift in chemical potential from neutral value; start of range (default to 0)
	const double dmuMax = inputMap.get("dmuMax", 0.) * eV; //optional shift in chemical potential from neutral value; end of range (default to 0)
	const int dmuCount = inputMap.get("dmuCount", 1); assert(dmuCount>0); //number of chemical potential shifts
	const double broadening = inputMap.get("broadening", 0)*eV; //if non-zero, override broadening with this value in eV
	const bool emission = inputMap.getBool("emission", false); //If true, also compute emission distributions
	string contribution = inputMap.getString("contribution"); //direct / phonon
	FeynWannParams fwp(&inputMap);
	
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
	logPrintf("polRe = "); polRe.print(globalLog, " %lg ");
	logPrintf("polIm = "); polIm.print(globalLog, " %lg ");
	logPrintf("GammaS = %lg\n", GammaS);
	logPrintf("eta = %lg\n", eta);
	logPrintf("dmuMin = %lg\n", dmuMin);
	logPrintf("dmuMax = %lg\n", dmuMax);
	logPrintf("dmuCount = %d\n", dmuCount);
	logPrintf("broadening = %lg\n", broadening);
	logPrintf("emission = %s\n", (emission ? "yes" : "no"));
	logPrintf("contribution = %s\n", contribMap.getString(contribType));
	fwp.printParams();
	
	//Initialize FeynWann:
	bool needLW = (broadening==0.);
	fwp.needPhonons = (contribType==Phonon);
	fwp.needVelocity = true;
	fwp.needLinewidth_ee = needLW;
	fwp.needLinewidth_ePh = needLW;
	std::shared_ptr<FeynWann> fw = std::make_shared<FeynWann>(fwp);
	size_t nKeff = nOffsets * (contribType==Direct ? fw->eCountPerOffset() : fw->ePhCountPerOffset());
	logPrintf("Effectively sampled %s: %lu\n", (contribType==Direct ? "nKpts" : "nKpairs"), nKeff);
	if(mpiWorld->isHead())
		logPrintf("%d %s-mesh offsets parallelized over %d process groups.\n",
			nOffsets, (contribType==Direct ? "electron k" : "phonon q"), mpiGroupHead->nProcesses());
	
	//Calculate normalized polarization vector:
	vector3<complex> Ehat = complex(1,0)*polRe + complex(0,1)*polIm; //Efield direction
	Ehat *= (1./sqrt(Ehat[0].norm() + Ehat[1].norm() + Ehat[2].norm())); //normalize to unit complex vector
	logPrintf("Ehat: [ %lg+%lgi , %lg+%lgi , %lg+%lgi ]\n", Ehat[0].real(), Ehat[0].imag(),
		Ehat[1].real(), Ehat[1].imag(), Ehat[2].real(), Ehat[2].imag() );
	
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
	EnergyRange er = { DBL_MAX, -DBL_MAX, 0. };
	for(vector3<> qOff: fw->qOffset) fw->eLoop(qOff, EnergyRange::eProcess, &er);
	mpiWorld->allReduce(er.Emin, MPIUtil::ReduceMin);
	mpiWorld->allReduce(er.Emax, MPIUtil::ReduceMax);
	mpiWorld->allReduce(er.vMax, MPIUtil::ReduceMax);
	double omegaFull = er.Emax - er.Emin;
	double vMax = 1.1*er.vMax; //add some margin
	double dv = 0.01; //in atomic units (since typical vF is 0.5 - 1)
	
	//dmu array:
	std::vector<double> dmu(dmuCount, dmuMin); //set first value here
	for(int iMu=1; iMu<dmuCount; iMu++) //set remaining values (if any)
		dmu[iMu] = dmuMin + iMu*(dmuMax-dmuMin)/(dmuCount-1);
	
	//Calculate delta-function resolved versions (no broadening yet):
	CollectImEps cie(dmu, T, domega, omegaFull, omegaMax, needLW, dv, vMax, emission);
	cie.prefac = 4. * std::pow(M_PI,2) * fw->spinWeight / (nKeff*fabs(det(fw->R))); //frequency independent part of prefactor
	cie.eta = eta;
	cie.GammaS = GammaS;
	cie.Ehat = Ehat;
	
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
	for(int iMu=0; iMu<dmuCount; iMu++)
	{	cie.ImEps[iMu].allReduce(MPIUtil::ReduceSum);
		if(needLW)
		{	cie.breadth[iMu].allReduce(MPIUtil::ReduceSum);
			if(contribType==Direct)
				cie.breadthDen[iMu] = cie.ImEps[iMu]; //normalization weight is just ImEps
			else
				cie.breadthDen[iMu].allReduce(MPIUtil::ReduceSum); //collected separately due to extrapolation sign
		}
	}
	cie.ImEps_E.allReduce(MPIUtil::ReduceSum);
	cie.ImEps_v.allReduce(MPIUtil::ReduceSum);
	if(emission)
	{	cie.Emission_E->allReduce(MPIUtil::ReduceSum);
		cie.Emission_E->print("emission"+fileSuffix+".dat", 1./eV, 1./eV, 1.);
		//Self-emission part for direct contributions:
		if(contribType == Direct)
		{	cie.selfEmission_E->allReduce(MPIUtil::ReduceSum);
			cie.selfEmission_E->print("selfEmission"+fileSuffix+".dat", 1./eV, 1./eV, 1.);
		}
	}
	logPrintf("done.\n"); logFlush();
	
	//Normalize the breadths:
	int nomega = cie.breadth[0].nE;
	for(int iomega=0; iomega<nomega; iomega++)
		for(int iMu=0; iMu<dmuCount; iMu++)
		{	cie.breadth[iMu].out[iomega] = needLW
				? std::max(T, cie.breadthDen[iMu].out[iomega]
					? cie.breadth[iMu].out[iomega]/cie.breadthDen[iMu].out[iomega]
					: 0.)
				: broadening; //use input value
		}
	cie.breadth[0].print("breadth"+fileSuffix+".dat", 1./eV, 1./eV);
	
	//Apply broadening:
	std::vector<Histogram> ImEps(dmuCount, Histogram(0, domega, omegaFull));
	Histogram2D ImEps_E(-omegaMax, domega, omegaMax,  0, domega, omegaMax);
	Histogram2D ImEps_v(-vMax, dv, vMax,  0, domega, omegaMax);
	int iomegaStart, iomegaStop; TaskDivision(nomega, mpiWorld).myRange(iomegaStart, iomegaStop);
	logPrintf("Applying broadening ... "); logFlush();
	for(int iMu=0; iMu<dmuCount; iMu++)
	{	for(int iomega=iomegaStart; iomega<iomegaStop; iomega++) //input frequency grid split over MPI
		{	double omegaCur = iomega*domega;
			double b = cie.breadth[iMu].out[iomega];
			for(size_t jomega=0; jomega<ImEps[iMu].out.size(); jomega++) //output frequency grid
			{	double omega = jomega*domega;
				double kernel = (needLW ? lorentzianOdd(omega, omegaCur, b) : gaussianOdd(omega, omegaCur, b)) * domega;
				ImEps[iMu].out[jomega] += kernel * cie.ImEps[iMu].out[iomega];
				//Carrier energy / speed distributions:
				if(iMu==0 && int(jomega)<ImEps_E.nomega)
				{	const int nE = ImEps_E.nE; assert(nE == cie.ImEps_E.nE);
					for(int iE=0; iE<nE; iE++)
					{	int iOE = iomega*nE + iE;
						int jOE = jomega*nE + iE;
						ImEps_E.out[jOE] += kernel * cie.ImEps_E.out[iOE];
					}
					const int nv = ImEps_v.nE; assert(nv == cie.ImEps_v.nE);
					for(int iv=0; iv<nv; iv++)
					{	int iOv = iomega*nv + iv;
						int jOv = jomega*nv + iv;
						ImEps_v.out[jOv] += kernel * cie.ImEps_v.out[iOv];
					}
				}
			}
		}
		ImEps[iMu].allReduce(MPIUtil::ReduceSum);
	}
	ImEps_E.allReduce(MPIUtil::ReduceSum); ImEps_E.print("carrierDistrib"+fileSuffix+".dat", 1./eV, 1./eV, 1.);
	ImEps_v.allReduce(MPIUtil::ReduceSum); ImEps_v.print("velocityDistrib"+fileSuffix+".dat", 1., 1./eV, 1.);
	logPrintf("done.\n"); logFlush();
	
	//Output ImEps:
	if(mpiWorld->isHead())
	{	ofstream ofs("ImEps"+fileSuffix+".dat");
		ofs << "#omega[eV]";
		for(int iMu=0; iMu<dmuCount; iMu++)
			ofs << " ImEps[mu=" << dmu[iMu]/eV << "eV]";
		ofs << "\n";
		for(size_t iOmega=0; iOmega<ImEps[0].out.size(); iOmega++)
		{	double omega = ImEps[0].Emin + ImEps[0].dE * iOmega;
			ofs << omega/eV;
			for(int iMu=0; iMu<dmuCount; iMu++)
				ofs << '\t' << ImEps[iMu].out[iOmega];
			ofs << '\n';
		}
	}
	
	fw = 0;
	FeynWann::finalize();
	return 0;
}
