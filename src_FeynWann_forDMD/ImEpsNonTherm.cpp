/*-------------------------------------------------------------------
Copyright 2018 Adela Habib

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
#include <core/Units.h>
#include "FeynWann.h"
#include "Histogram.h"
#include "InputMap.h"
#include "Interp1.h"

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
{	const std::vector<double>& Tl;
	int numTimes;
	double GammaS;
	double Tl0lw;
	double dE, omegaFull;
	std::vector<Histogram> ImEps, breadth, breadthDen;
	const Interp1 &fInterp, &lwInterp;
	double prefac;
	double eta; //singularity extrapolation width
	vector3<> Ehat;
	
	CollectImEps(const std::vector<double>& Tl, double dE, double omegaFull, const Interp1& fInterp, const Interp1& lwInterp)
	: Tl(Tl), numTimes(Tl.size()), dE(dE), omegaFull(omegaFull), 
		ImEps(numTimes, Histogram(0, dE, omegaFull)),
		breadth(numTimes, Histogram(0, dE, omegaFull)),
		breadthDen(numTimes, Histogram(0, dE, omegaFull)),
		fInterp(fInterp), lwInterp(lwInterp)
	{	logPrintf("Initialized frequency grid: 0 to %lg eV with %d points.\n", ImEps[0].Emax()/eV, ImEps[0].nE);
		
	}
	
	void calcStateRelated(const FeynWann::StateE& state, std::vector<diagMatrix>& F, std::vector<diagMatrix>& ImE)
	{	int nBands = state.E.nRows();
		F.assign(nBands, diagMatrix(numTimes));
		ImE.assign(nBands, diagMatrix(numTimes));
		for(int b=0; b<nBands; b++)
		{	double E = state.E[b];
			for(int iT=0; iT<numTimes; iT++)
			{	double TlRatio = Tl[iT] / Tl0lw;
				F[b][iT] = fInterp(iT,E); //interpolate electron occupations (not necessarily a Fermi distribution)
				ImE[b][iT] = state.ImSigma_ee[b] //e-e contribution
					+ state.ImSigma_ePh(b, F[b][iT])*TlRatio //e-ph contribution with linear Tl dependence
					+ lwInterp(iT, E); //add linewidth correction (interpolated as a function of carrier energy)
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
				if(omega<dE || omega>=omegaFull) continue; //irrelevant event
				double weight_F = (prefac/(omega*omega)) * P(c,v).norm(); //event weight except for occupation factors
				//Collect results:
				int iOmega; double tOmega; //coordinates of frequency on frequency grid
				bool useEvent = ImEps[0].eventPrecalc(omega, iOmega, tOmega); //all histograms on same frequency grid
				if(useEvent) for(int iT=0; iT<numTimes; iT++)
				{	double weight = weight_F * (F[v][iT]-F[c][iT]);
					ImEps[iT].addEventPrecalc(iOmega, tOmega, weight);
					breadth[iT].addEventPrecalc(iOmega, tOmega, weight*(ImE[c][iT]+ImE[v][iT]+GammaS));
				}
			}
		}
	}
	static void direct(const FeynWann::StateE& state, void* params)
	{	((CollectImEps*)params)->collectDirect(state);
	}
	
	//---- Phonon-assisted transitions ----
	void collectPhonon(const FeynWann::MatrixEph& mat)
	{	static StopWatch watchME("cPh:MatElem"), watchHist("cPh:Hist");
		int nBands = mat.e1->E.nRows();
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
		
		std::vector<diagMatrix> nPh(nModes, diagMatrix(numTimes));
		for(int iMode=0; iMode<nModes; iMode++)
			for(int iT=0; iT<numTimes; iT++)
			{	double omegaPhByT = omegaPh[iMode]/Tl[iT];
				nPh[iMode][iT] = bose(std::max(1e-3, omegaPhByT)); //avoid 0/0 for zero phonon frequencies
			}
		
		//Collect
		for(int v=0; v<nBands; v++) 
		{	double* F1v = F1[v].data();
			double* ImE1v = ImE1[v].data();
			for(int c=0; c<nBands; c++) 
			{	double* F2c = F2[c].data();
				double* ImE2c = ImE2[c].data();
				for(int alpha=0; alpha<nModes; alpha++)
				{	double* nPhAlpha = nPh[alpha].data();
					for(int ae=-1; ae<=+1; ae+=2) // +/- for phonon absorption or emmision
					{	double omega = E2[c] - E1[v] - ae*omegaPh[alpha]; //energy conservation
						if(omega<dE || omega>=omegaFull) continue; //irrelevant event
						//Effective matrix elements
						watchME.start();
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
						double weight_occ =  (prefac/(omega*omega)) * MeffSqExtrap; //event weight, except for occupation factors
						watchME.stop();
						//Collect results:
						watchHist.start();
						int iOmega; double tOmega; //coordinates of frequency on frequency grid
						bool useEvent = ImEps[0].eventPrecalc(omega, iOmega, tOmega); //all histograms on same frequency grid
						if(useEvent) for(int iT=0; iT<numTimes; iT++)
						{	double weight = weight_occ * (nPhAlpha[iT] + 0.5*(1.-ae)) * (F1v[iT]-F2c[iT]);
							ImEps[iT].addEventPrecalc(iOmega, tOmega, weight);
							breadth[iT].addEventPrecalc(iOmega, tOmega, fabs(weight)*(ImE2c[iT]+ImE1v[iT]+GammaS));
							breadthDen[iT].addEventPrecalc(iOmega, tOmega, fabs(weight));
						}
						watchHist.stop();
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

inline void writeImEps(const char* fname, const std::vector<Histogram>& ImEps, const std::vector<string>& headerVals)
{	std::ofstream ofs(fname);
	//Header:
	ofs << "#omega[eV]";
	for(const string& headerVal: headerVals)
		ofs << ' ' << headerVal;
	ofs << '\n';
	//Data:
	for(size_t iomega=0; iomega<ImEps[0].out.size(); iomega++)
	{	double omega = ImEps[0].dE * iomega;
		ofs << omega/eV;
		for(const Histogram& h: ImEps)
			ofs << ' ' << h.out[iomega];
		ofs << '\n';
	}
}

int main(int argc, char** argv)
{	
	InitParams ip = FeynWann::initialize(argc, argv, "Wannier calculation of imaginary dielectric tensor (ImEps)");

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(ip.inputFilename);
	const int nOffsets = inputMap.get("nOffsets"); assert(nOffsets>0);
	const double dE = inputMap.get("dE") * eV; //energy resolution used for output and energy conservation
	const double GammaS = inputMap.get("GammaS", 0.) * eV; //extra broadening due to surface scattering (default: none)
	string runName = inputMap.getString("runName");
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
	logPrintf("dE = %lg\n", dE);
	logPrintf("GammaS = %lg\n", GammaS);
	logPrintf("runName = %s\n", runName.c_str());
	logPrintf("contribution = %s\n", contribMap.getString(contribType));
	fwp.printParams();
	
	//Initialize FeynWann:
	fwp.needPhonons = (contribType==Phonon);
	fwp.needVelocity = true;
	fwp.needLinewidth_ee = true;
	fwp.needLinewidth_ePh = true;
	std::shared_ptr<FeynWann> fw = std::make_shared<FeynWann>(fwp);
	size_t nKeff = nOffsets * (contribType==Direct ? fw->eCountPerOffset() : fw->ePhCountPerOffset());
	logPrintf("Effectively sampled %s: %lu\n", (contribType==Direct ? "nKpts" : "nKpairs"), nKeff);
	if(mpiWorld->isHead())
		logPrintf("%d %s-mesh offsets parallelized over %d process groups.\n",
			nOffsets, (contribType==Direct ? "electron k" : "phonon q"), mpiGroupHead->nProcesses());

	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		fw = 0;
		FeynWann::finalize();
		return 0;
	}
	logPrintf("\n");
	
	Interp1 fInterp, lwInterp;
	fInterp.init(runName + ".f", eV, 1.);
	lwInterp.init(runName + ".lwDelta", eV, eV);
	int numTimes = fInterp.headerVals.size();
	//Read lattice temperatures:
	Interp1 TlInterp; TlInterp.init(runName + ".Tl", fs, Kelvin);
	const std::vector<double>& Tl = TlInterp.yGrid[0];
	assert(int(Tl.size()) == numTimes);
	//Initialize sampling parameters:
	int oStart=0, oStop=0;
	if(mpiGroup->isHead())
		TaskDivision(nOffsets, mpiGroupHead).myRange(oStart, oStop);
	mpiGroup->bcast(oStart);
	mpiGroup->bcast(oStop);
	int noMine = oStop-oStart; //number of offsets handled by current group
	int oInterval = std::max(1, int(round(noMine/50.))); //interval for reporting progress
	//Initialize frequency grid:
	EnergyRange er = { DBL_MAX, -DBL_MAX };
	for(vector3<> qOff: fw->qOffset) fw->eLoop(qOff, EnergyRange::eProcess, &er);
	mpiWorld->allReduce(er.Emin, MPIUtil::ReduceMin);
	mpiWorld->allReduce(er.Emax, MPIUtil::ReduceMax);
	double omegaFull = er.Emax - er.Emin;
	
	//Calculate delta-function resolved versions (no broadening yet):
	CollectImEps cie(Tl, dE, omegaFull, fInterp, lwInterp);
	cie.prefac = 4. * std::pow(M_PI,2) * fw->spinWeight / (nKeff*fabs(det(fw->R))); //frequency independent part of prefactor
	cie.Ehat = vector3<>(1., 0., 0.);  //assume cubic symmetry and only calculate x-axis
	cie.eta = 0.1*eV;
	cie.Tl0lw = 0.026*eV;
	cie.GammaS = GammaS;
	
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
	for(int iT=0; iT<numTimes; iT++)
	{	cie.ImEps[iT].allReduce(MPIUtil::ReduceSum);
		cie.breadth[iT].allReduce(MPIUtil::ReduceSum);
		if(contribType==Direct)
			cie.breadthDen[iT] = cie.ImEps[iT]; //normalization weight is just ImEps
		else
			cie.breadthDen[iT].allReduce(MPIUtil::ReduceSum); //collected separately due to extrapolation sign	
	}
	logPrintf("done.\n"); logFlush();
	
	//Normalize the breadths:
	int nomega = cie.breadth[0].nE;
	for(int iomega=0; iomega<nomega; iomega++)
	{	for(int iT=0; iT<numTimes; iT++)
		{	cie.breadth[iT].out[iomega] = std::max(dE, 
				cie.breadthDen[iT].out[iomega]
					? cie.breadth[iT].out[iomega]/cie.breadthDen[iT].out[iomega]
					: 0.);
		}
	}
	
	//Apply broadening:
	std::vector<Histogram> ImEps(numTimes, Histogram(0, dE, omegaFull));
	int iomegaStart, iomegaStop; TaskDivision(nomega, mpiWorld).myRange(iomegaStart, iomegaStop);
	logPrintf("Applying broadening ... "); logFlush();
	for(int iT=0; iT<numTimes; iT++)
	{	for(int iomega=iomegaStart; iomega<iomegaStop; iomega++) //input frequency grid split over MPI
		{	double omegaCur = iomega*dE;
			double b = cie.breadth[iT].out[iomega];
			for(size_t jomega=0; jomega<ImEps[iT].out.size(); jomega++) //output frequency grid
			{	double omega = jomega*dE;
				double kernel = lorentzianOdd(omega, omegaCur, b) * dE;
				ImEps[iT].out[jomega] += kernel * cie.ImEps[iT].out[iomega];
			}
		}
		ImEps[iT].allReduce(MPIUtil::ReduceSum);
	}
	logPrintf("done.\n"); logFlush();
	
	//Output ImEps:
	if(mpiWorld->isHead())
	{	string fName1 = "ImEps_"+fileSuffix+"Nontherm_"+runName+".dat";
		writeImEps(fName1.c_str(), ImEps, fInterp.headerVals);
		
		//Print Linewidth correction at each time:
		string fName3 = "LWcorrection_"+runName+".dat";
		ofstream ofs(fName3);
		ofs << "#LinewidthCorrection[eV]\n";
		for(int iT=0; iT<numTimes; iT++)
		{	ofs << lwInterp(iT,0)/eV << '\n';
		}
	}
	fw = 0;
	FeynWann::finalize();
	return 0;
}
