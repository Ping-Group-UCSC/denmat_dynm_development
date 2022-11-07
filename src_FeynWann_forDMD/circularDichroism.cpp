/*-------------------------------------------------------------------
Copyright 2019 Ravishankar Sundararaman

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
#include <core/tensor3.h>
#include <core/Random.h>
#include <core/string.h>
#include "FeynWann.h"
#include "Histogram.h"
#include "InputMap.h"
#include <core/Units.h>

//Levi-civita operators:
template<typename scalar> matrix3<scalar> epsDot(const vector3<scalar>& v)
{	matrix3<scalar> M;
	M(2,1) = -(M(1,2) = v[0]);
	M(0,2) = -(M(2,0) = v[1]);
	M(1,0) = -(M(0,1) = v[2]);
	return M;
}
template<typename scalar> vector3<scalar> epsDot(const matrix3<scalar>& M)
{	return vector3<scalar>(
		M(1,2) - M(2,1),
		M(2,0) - M(0,2),
		M(0,1) - M(1,0) );
}
template<typename scalar> matrix3<scalar> Sym(const matrix3<scalar>& M)
{	return scalar(0.5)*(M + (~M));
}
inline matrix3<> Real(const matrix3<complex>& M)
{	matrix3<> ret;
	for(int i=0; i<3; i++)
		for(int j=0; j<3; j++)
			ret(i,j) = M(i,j).real();
	return ret;
}

inline vector3<complex> conj(const vector3<complex>& v)
{	return vector3<complex>(v[0].conj(), v[1].conj(), v[2].conj());
}

inline vector3<complex> getVectorElement(const matrix M[3], int b1, int b2)
{	vector3<complex> result;
	for(int iDir=0; iDir<3; iDir++)
		result[iDir] = M[iDir](b1, b2);
	return result;
}

inline tensor3<complex> getTensorElement(const matrix M[5], int b1, int b2)
{	tensor3<complex> result;
	for(int iComp=0; iComp<5; iComp++)
		result[iComp] = M[iComp](b1, b2);
	return result;
}


//Singularity extrapolation for phonon-assisted:
double extrapCoeff[] = {-19./12, 13./3, -7./4 }; //account for constant, 1/eta and eta^2 dependence
//double extrapCoeff[] = { -1, 2.}; //account for constant and 1/eta dependence
const int nExtrap = sizeof(extrapCoeff)/sizeof(double);


//Collect circular dichroism contibutions using FeynWann callbacks:
struct CollectCD
{	double dmu, T, invT;
	double domega, omegaMax;
	bool spinAvailable;
	std::vector<Histogram> CD, CDmd; //Total circular dichorism and magnetic momentum contributions alone (xx,yy,zz,yz,zx,xy components)
	std::vector<Histogram> CDspin; //Spin contributions, if present
	const matrix3<> Id; //3x3 identity
	double prefac;
	double eta; //singularity extrapolation width

	CollectCD(double dmu, double T, double domega, double omegaMax, bool spinAvailable)
	: dmu(dmu), T(T), invT(1./T), domega(domega), omegaMax(omegaMax), spinAvailable(spinAvailable),
		CD(6, Histogram(0, domega, omegaMax)),
		CDmd(6, Histogram(0, domega, omegaMax)),
		Id(1.,1.,1.)
	{	logPrintf("Initialized frequency grid: 0 to %lg eV with %d points.\n", CD[0].Emax()/eV, CD[0].nE);
		if(spinAvailable) CDspin.assign(6, Histogram(0, domega, omegaMax));
	}
	
	diagMatrix getFillings(const diagMatrix& E)
	{	diagMatrix F(E);
		for(double& e: F)
			e = fermi((e-dmu)*invT); //convert energies to fillings in place
		return F;
	}
	
	//Common logic between direct and phonon-assisted CD calculations
	inline void collectCommon(double omega, double weight,
		const vector3<complex>& P, const tensor3<complex>& Q, const vector3<complex>& L, const vector3<complex>& Sdbl)
	{	//Compute contributions:
		vector3<complex> Pconj = conj(P);
		matrix3<> Feq = Sym(Real(epsDot(Pconj) * matrix3<complex>(Q)));
		matrix3<> Fmd = Id*dot(Pconj, L).real() - Sym(Real(outer(Pconj, L))), Fspin;
		if(spinAvailable) Fspin = Id*dot(Pconj, Sdbl).real() - Sym(Real(outer(Pconj, Sdbl)));
		matrix3<> Ftot = Feq + Fmd + Fspin;
		//Save contribution to appropriate frequency:
		int iOmega; double tOmega; //coordinates of frequency on frequency grid
		bool useEvent = CD[0].eventPrecalc(omega, iOmega, tOmega); //all histograms on same frequency grid
		if(useEvent)
		{
			#define addEventTensor(H, G) \
				H[0].addEventPrecalc(iOmega, tOmega, weight*G(0,0)); \
				H[1].addEventPrecalc(iOmega, tOmega, weight*G(1,1)); \
				H[2].addEventPrecalc(iOmega, tOmega, weight*G(2,2)); \
				H[3].addEventPrecalc(iOmega, tOmega, weight*G(1,2)); \
				H[4].addEventPrecalc(iOmega, tOmega, weight*G(2,0)); \
				H[5].addEventPrecalc(iOmega, tOmega, weight*G(0,1));
			addEventTensor(CD, Ftot);
			addEventTensor(CDmd, Fmd);
			if(spinAvailable) { addEventTensor(CDspin, Fspin); }
			#undef addEventTensor
		}
	}
	
	//---- Direct transitions ----
	void collectDirect(const FeynWann::StateE& state)
	{	int nBands = state.E.nRows();
		//Calculate Fermi fillings and linewidths:
		const diagMatrix &E = state.E, F = getFillings(E);
		//Collect 
		for(int b2=0; b2<nBands; b2++) //initial electronic band (n in derivation)
		{	for(int b1=0; b1<nBands; b1++) //final electronic band (n' in derivation)
			{	double omega = E[b1] - E[b2]; //energy conservation
				double df = F[b2] - F[b1]; 
				if(omega<domega || omega>=omegaMax) continue; //irrelevant event
				if(df < 1E-6) continue; //negligible weight below
				//Collect relevant matrix elements:
				vector3<complex> P = getVectorElement(state.v, b1, b2);
				vector3<complex> L = getVectorElement(state.L, b1, b2), Sdbl;
				if(spinAvailable) Sdbl = getVectorElement(state.S, b1, b2); //2*S
				tensor3<complex> Q(getTensorElement(state.Q, b1, b2));
				//Collect contributions:
				collectCommon(omega, prefac * df, P, Q, L, Sdbl);
			}
		}
	}
	static void direct(const FeynWann::StateE& state, void* params)
	{	((CollectCD*)params)->collectDirect(state);
	}
	
	//---- Phonon-assisted transitions ----
	void collectPhonon(const FeynWann::MatrixEph& mat)
	{	int nBands = mat.e1->E.nRows();
		//Calculate Fermi fillings and linewidths:
		const diagMatrix& E1 = mat.e1->E, F1 = getFillings(E1); //properties at k1 (k' in derivation)
		const diagMatrix& E2 = mat.e2->E, F2 = getFillings(E2); //properties at k2 (k in derivation)
		//Bose occupations:
		const diagMatrix& omegaPh = mat.ph->omega;
		int nModes = omegaPh.nRows();
		diagMatrix nPh(nModes);
		for(int iMode=0; iMode<nModes; iMode++)
		{	double omegaPhByT = omegaPh[iMode]/T;
			nPh[iMode] = bose(std::max(1e-3, omegaPhByT)); //avoid 0/0 for zero phonon frequencies
		}
		//Collect
		for(int b2=0; b2<nBands; b2++) //initial electron band (n in derivation)
		{	for(int b1=0; b1<nBands; b1++) //final electron band (n' in derivation)
			{	double df = F2[b2] - F1[b1]; 
				if(df < 1E-6) continue; //negligible weight below
				for(int alpha=0; alpha<nModes; alpha++) //phonon mode
				{	for(int ae=-1; ae<=+1; ae+=2) // +/- for phonon absorption or emmision
					{	double omega = E1[b1] - E2[b2] - ae*omegaPh[alpha]; //energy conservation
						if(omega<domega || omega>=omegaMax) continue; //irrelevant event
						//Effective matrix elements in second order perturbation theory:
						std::vector<vector3<complex>> P(nExtrap), L(nExtrap), Sdbl(nExtrap);
						std::vector<tensor3<complex>> Q(nExtrap);
						for(int i=0; i<nBands; i++) //sum over the intermediate states (n1 in derivation)
						{	//Relevant e-ph matrix elements:
							complex g_1i = mat.M[alpha](b1, i);
							complex g_i2 = mat.M[alpha](i, b2);
							//Corresponding energy denominator real parts:
							double denA = E2[i] - E2[b2] - omega;
							double denB = E1[i] - E1[b1] + omega;
							//Raw optical coupling matrix elements:
							vector3<complex> P_1i = getVectorElement(mat.e1->v, b1, i);
							vector3<complex> P_i2 = getVectorElement(mat.e2->v, i, b2);
							vector3<complex> L_1i = getVectorElement(mat.e1->L, b1, i), Sdbl_1i;
							vector3<complex> L_i2 = getVectorElement(mat.e2->L, i, b2), Sdbl_i2;
							if(spinAvailable)
							{	Sdbl_1i = getVectorElement(mat.e1->S, b1, i); //2S
								Sdbl_i2 = getVectorElement(mat.e2->S, i, b2); //2S
							}
							tensor3<complex> Q_1i = getTensorElement(mat.e1->Q, b1, i);
							tensor3<complex> Q_i2 = getTensorElement(mat.e2->Q, i, b2);
							//Loop over energy denominator imaginary parts (for extrapolation):
							double zEta = eta;
							for(int z=0; z<nExtrap; z++)
							{	complex g_1i_den = g_1i / complex(denA, zEta);
								complex g_i2_den = g_i2 / complex(denB, zEta);
								zEta += eta; //contains (z+1)*eta when evaluating above
								//Accumulate effective optical matrix elements:
								P[z] += g_1i_den * P_i2 + P_1i * g_i2_den;
								L[z] += g_1i_den * L_i2 + L_1i * g_i2_den;
								if(spinAvailable) Sdbl[z] += g_1i_den * Sdbl_i2 + Sdbl_1i * g_i2_den;
								Q[z] += g_1i_den * Q_i2 + Q_1i * g_i2_den;
							}
						}
						//Collect with singularity extrapolation:
						double weight = prefac * df * (nPh[alpha] + 0.5*(1.-ae));
						for(int z=0; z<nExtrap; z++)
							collectCommon(omega, weight * extrapCoeff[z], P[z], Q[z], L[z], Sdbl[z]);
					}
				}
			}
		}
	}
	static void phonon(const FeynWann::MatrixEph& mat, void* params)
	{	((CollectCD*)params)->collectPhonon(mat);
	}
	
	void allReduce()
	{	for(Histogram& h: CD) h.allReduce(MPIUtil::ReduceSum);
		for(Histogram& h: CDmd) h.allReduce(MPIUtil::ReduceSum);
		if(spinAvailable) for(Histogram& h: CDspin) h.allReduce(MPIUtil::ReduceSum);
	}
	
	void saveTensor(const std::vector<Histogram>& hArr, string fname, const FeynWann& fw)
	{	if(mpiWorld->isHead())
		{	ofstream ofs(fname.c_str());
			//Header:
			ofs << "#omega[eV]";
			const char* comps[6] = { "xx", "yy", "zz", "yz", "zx", "xy" };
			for(int iComp=0; iComp<6; iComp++)
				ofs << " dAlpha_" << comps[iComp] << "[cm^-1]";
			ofs << "\n";
			//Result for each frequency in a row:
			for(size_t iOmega=0; iOmega<hArr[0].out.size(); iOmega++)
			{	double omega = hArr[0].Emin + hArr[0].dE * iOmega;
				ofs << omega/eV;
				//Collect and symmetrize tensor:
				matrix3<> M;
				M(0,0) = hArr[0].out[iOmega];
				M(1,1) = hArr[1].out[iOmega];
				M(2,2) = hArr[2].out[iOmega];
				M(1,2) = (M(2,1) = hArr[3].out[iOmega]);
				M(2,0) = (M(0,2) = hArr[4].out[iOmega]);
				M(0,1) = (M(1,0) = hArr[5].out[iOmega]);
				fw.symmetrize(M);
				//Switch units:
				M *= (1e8*Angstrom); //switch from atomic units to cm^-1
				//Write components:
				ofs << '\t' << M(0,0) << '\t' << M(1,1) << '\t' << M(2,2)
					<< '\t' << M(1,2) << '\t' << M(2,0) << '\t' << M(0,1) << '\n';
			}
		}
	}
	void save(const FeynWann& fw, string fileSuffix)
	{	saveTensor(CD, "CD" + fileSuffix + ".dat", fw);
		saveTensor(CDmd, "CDmd" + fileSuffix + ".dat", fw);
		if(spinAvailable) saveTensor(CDspin, "CDspin" + fileSuffix + ".dat", fw);
	}
};

int main(int argc, char** argv)
{	
	InitParams ip = FeynWann::initialize(argc, argv, "Wannier calculation of circular dichroism");

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(ip.inputFilename);
	const int nOffsets = inputMap.get("nOffsets"); assert(nOffsets>0);
	const double omegaMax = inputMap.get("omegaMax") * eV; assert(omegaMax>0.); //maximum photon frequency to collect results for
	const double domega = inputMap.get("domega") * eV; assert(domega>0.); //photon energy grid resolution
	const double T = inputMap.get("T") * Kelvin;
	const double dmu = inputMap.get("dmu", 0.) * eV; //optional shift in chemical potential from neutral value/ VBM; (default to 0)
	const double eta = inputMap.get("eta", 0.1) * eV; //on-shell extrapolation width (default to 0.1 eV)
string contribution = inputMap.has("contribution") ? inputMap.getString("contribution") : "direct"; //direct / phonon
	FeynWannParams fwp(&inputMap);

	//Check contribution:
	enum ContribType { Direct, Phonon };
	EnumStringMap<ContribType> contribMap(Direct, "Direct", Phonon, "Phonon");
	ContribType contribType;
	if(!contribMap.getEnum(contribution.c_str(), contribType))
		die("Input parameter 'contribution' must be one of %s.\n\n", contribMap.optionList().c_str());
	string fileSuffix = (contribType==Phonon) ? "_ph" : "";

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nOffsets = %d\n", nOffsets);
	logPrintf("omegaMax = %lg\n", omegaMax);
	logPrintf("domega = %lg\n", domega);
	logPrintf("T = %lg\n", T);
	logPrintf("dmu = %lg\n", dmu);
	logPrintf("eta = %lg\n", eta);
	logPrintf("contribution = %s\n", contribMap.getString(contribType));
	fwp.printParams();
	
	//Initialize FeynWann:
	fwp.needSymmetries = true;
	fwp.needVelocity = true;
	fwp.needPhonons = (contribType==Phonon);
	fwp.needQ = true;
	fwp.needL = true;
	fwp.needSpin = true; //for spin contribution, if available
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

	//Initialize sampling parameters:
	int oStart=0, oStop=0;
	if(mpiGroup->isHead())
		TaskDivision(nOffsets, mpiGroupHead).myRange(oStart, oStop);
	mpiGroup->bcast(oStart);
	mpiGroup->bcast(oStop);
	int noMine = oStop-oStart; //number of offsets handled by current group
	int oInterval = std::max(1, int(round(noMine/50.))); //interval for reporting progress
	
	//Collect results:
	CollectCD ccd(dmu, T, domega, omegaMax, fwp.needSpin);
	const double c = 137.035999084; //speed of light in atomic units = 1/(fine structure constant)
	ccd.prefac = 4.*std::pow(M_PI/c,2) * fw->spinWeight / (nKeff*fabs(det(fw->R))); //frequency independent part of prefactor
	ccd.eta = eta;
	
	for(int iSpin=0; iSpin<fw->nSpins; iSpin++)
	{	//Update FeynWann for spin channel if necessary:
		if(iSpin>0)
		{	fw = 0; //free memory from previous spin
			fwp.iSpin = iSpin;
			fw = std::make_shared<FeynWann>(fwp);
		}
		logPrintf("\nCollecting CD spectrum: "); logFlush();
		for(int o=0; o<noMine; o++)
		{	Random::seed(o+oStart); //to make results independent of MPI division
			//Process with a random offset:
			switch(contribType)
			{	case Direct:
				{	vector3<> k0 = fw->randomVector(mpiGroup); //must be constant across group
					fw->eLoop(k0, CollectCD::direct, &ccd);
					break;
				}
				case Phonon:
				{	vector3<> k01 = fw->randomVector(mpiGroup); //must be constant across group
					vector3<> k02 = fw->randomVector(mpiGroup); //must be constant across group
					fw->ePhLoop(k01, k02, CollectCD::phonon, &ccd);
					break;
				}
			}
			//Print progress:
			if((o+1)%oInterval==0) { logPrintf("%d%% ", int(round((o+1)*100./noMine))); logFlush(); }
		}
		logPrintf("done.\n"); logFlush();
	}
	ccd.allReduce();
	logPrintf("done.\n"); logFlush();
	
	//Output results:
	ccd.save(*fw, fileSuffix);
	
	fw = 0;
	FeynWann::finalize();
	return 0;
}
