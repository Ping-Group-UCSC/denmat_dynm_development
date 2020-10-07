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
#include <core/scalar.h>
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

//Collect circular dichroism contibutions using FeynWann callbacks:
struct CollectCD
{	double dmu, T, invT;
	double domega, omegaMax, EemptyMax;
	std::vector<Histogram> CD, CDmd; //Total circular dichorism and magnetic dipole contributions alone (xx,yy,zz,yz,zx,xy components)
	double prefac;
	
	CollectCD(double dmu, double T, double domega, double omegaMax, double EemptyMax)
	: dmu(dmu), T(T), invT(1./T), domega(domega), omegaMax(omegaMax), EemptyMax(EemptyMax),
		CD(6, Histogram(0, domega, omegaMax)),
		CDmd(6, Histogram(0, domega, omegaMax))
	{	logPrintf("Initialized frequency grid: 0 to %lg eV with %d points.\n", CD[0].Emax()/eV, CD[0].nE);
	}
	
	void collectE(const FeynWann::StateE& state)
	{	int nBands = state.E.nRows();
		const double degeneracyThreshold = 1e-4;
		matrix3<> Id(1.,1.,1.); //3x3 identity
		//Calculate Fermi fillings and linewidths:
		const diagMatrix& E = state.E;
		diagMatrix F(nBands);
		for(int b=0; b<nBands; b++)
			F[b] = fermi((state.E[b]-dmu)*invT);
		//Collect 
		for(int b2=0; b2<nBands; b2++)
		{	for(int b1=0; b1<nBands; b1++)
			{	double omega = E[b1] - E[b2]; //energy conservation
				if(omega<domega || omega>=omegaMax) continue; //irrelevant event
				//Get dipole matrix element:
				vector3<complex> P21;
				for(int iDir=0; iDir<3; iDir++)
					P21[iDir] = state.v[iDir](b1,b2);
				//Compute X12 = < r p >_12 by sum over empty states:
				matrix3<complex> X12;
				for(int b3=0; b3<nBands; b3++)
					if(E[b3]<EemptyMax and fabs(E[b3]-E[b2])>degeneracyThreshold)
					{	complex invDE23 = complex(0., 1./(E[b2]-E[b3]));
						vector3<complex> P13, r32;
						for(int iDir=0; iDir<3; iDir++)
						{	P13[iDir] = state.v[iDir](b3,b1);
							r32[iDir] = state.v[iDir](b2,b3) * invDE23;
						}
						X12 += outer(r32, P13);
					}
				//Compute EQ and MD contributions:
				//--- Magnetic Dipole (MD) contribution:
				vector3<complex> L12 = epsDot(X12);
				matrix3<> Gmd = Sym(Real(outer(P21,L12))) - Id*dot(P21,L12).real();
				//--- Electric Quadrupole (EQ) contribution:
				matrix3<> Geq = 2.*Sym(Real(Sym(X12) * epsDot(P21)));
				matrix3<> Gtot = Gmd + Geq;
				double weight = prefac * (F[b1] - F[b2]);
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
					addEventTensor(CD, Gtot);
					addEventTensor(CDmd, Gmd);
					#undef addEventTensor
				}
			}
		}
	}
	static void collect(const FeynWann::StateE& state, void* params)
	{	((CollectCD*)params)->collectE(state);
	}
	
	void allReduce()
	{	for(Histogram& h: CD) h.allReduce(MPIUtil::ReduceSum);
		for(Histogram& h: CDmd) h.allReduce(MPIUtil::ReduceSum);
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
	void save(const FeynWann& fw)
	{	saveTensor(CD, "CD.dat", fw);
		saveTensor(CDmd, "CDmd.dat", fw);
	}
};

int main(int argc, char** argv)
{	
	InitParams ip = FeynWann::initialize(argc, argv, "Wannier calculation of imaginary dielectric tensor (ImEps)");

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(ip.inputFilename);
	const int nOffsets = inputMap.get("nOffsets"); assert(nOffsets>0);
	const double omegaMax = inputMap.get("omegaMax") * eV; assert(omegaMax>0.); //maximum photon frequency to collect results for
	const double domega = inputMap.get("domega") * eV; assert(domega>0.); //photon energy grid resolution
	const double EemptyMax = inputMap.get("EemptyMax", +DBL_MAX) * eV; //maximum empty-state energy to use (vary below max Wannier energy) to check convergence
	const double T = inputMap.get("T") * Kelvin;
	const double dmu = inputMap.get("dmu", 0.) * eV; //optional shift in chemical potential from neutral value/ VBM; (default to 0)

	//Check contribution:
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nOffsets = %d\n", nOffsets);
	logPrintf("omegaMax = %lg\n", omegaMax);
	logPrintf("domega = %lg\n", domega);
	logPrintf("EemptyMax = %lg\n", EemptyMax);
	logPrintf("T = %lg\n", T);
	logPrintf("dmu = %lg\n", dmu);

	//Initialize FeynWann:
	FeynWannParams fwp;
	fwp.needSymmetries = true;
	fwp.needVelocity = true;
	std::shared_ptr<FeynWann> fw = std::make_shared<FeynWann>(fwp);
	size_t nKeff = nOffsets * fw->eCountPerOffset();
	logPrintf("Effectively sampled nKpts: %lu\n", nKeff);

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
	CollectCD ccd(dmu, T, domega, omegaMax, EemptyMax);
	const double c = 137.035999084; //speed of light in atomic units = 1/(fine structure constant)
	ccd.prefac = 4.*std::pow(M_PI/c,2) * fw->spinWeight / (nKeff*fabs(det(fw->R))); //frequency independent part of prefactor
	
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
			vector3<> k0 = fw->randomVector(mpiGroup); //must be constant across group
			fw->eLoop(k0, CollectCD::collect, &ccd);
			//Print progress:
			if((o+1)%oInterval==0) { logPrintf("%d%% ", int(round((o+1)*100./noMine))); logFlush(); }
		}
		logPrintf("done.\n"); logFlush();
	}
	ccd.allReduce();
	logPrintf("done.\n"); logFlush();
	
	//Output results:
	ccd.save(*fw);
	
	fw = 0;
	FeynWann::finalize();
	return 0;
}
