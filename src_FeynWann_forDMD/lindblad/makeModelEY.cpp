#include <core/matrix.h>
#include <core/Random.h>
#include <core/Units.h>
#include <InputMap.h>
#include <lindblad/LindbladFile.h>


int main(int argc, char** argv)
{
	InitParams ip = FeynWann::initialize(argc, argv, "Create an isotropic Elliott-Yafett model 2-band spin system");
	if(mpiWorld->nProcesses() > 1) die("MPI not supported for this quick model generation code.\n\n");

	//Get input parameters:
	InputMap inputMap(ip.inputFilename);
	const int nK = int(inputMap.get("nK")); //number of k-points
	const double sigmaL = inputMap.get("sigmaL"); //std. dev. of orbital angular momentum ~ std. dev. of g-factor
	const double tauP = inputMap.get("tauP") * fs; //target carrier/momentum relaxation time in fs
	const double tauS = inputMap.get("tauS") * fs; //target EY spin relaxation time (T1) in fs
	const double fracNZ = inputMap.get("fracNZ"); //fraction of non-zero scattering matrix elements (control sparsity)
	const int nBands = 2;
	
	//Print back input parameters (converted):
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nK = %d\n", nK);
	logPrintf("sigmaL = %lg\n", sigmaL);
	logPrintf("tauP = %lg\n", tauP);
	logPrintf("tauS = %lg\n", tauS);
	logPrintf("fracNZ = %lg\n", fracNZ);
	logPrintf("\n");
	
	//Compute spin mixing:
	const double spinMixSq = tauP / (4 * tauS); //b^2 in EY formula
	if(spinMixSq > 0.5) die("EY tauS must be >= 2 tauP.\n\n");
	
	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		FeynWann::finalize();
		return 0;
	}
	logPrintf("\n");
	
	//Create Pauli matrices:
	vector3<matrix> pauli;
	// sigma_x
	pauli[0] = zeroes(nBands, nBands);
	pauli[0].set(0,1, 1);
	pauli[0].set(1,0, 1);
	// sigma_y
	pauli[1] = zeroes(nBands, nBands);
	pauli[1].set(0,1, complex(0,-1));
	pauli[1].set(1,0, complex(0, 1));
	// sigma_z
	pauli[2] = zeroes(nBands, nBands);
	pauli[2].set(0,0,  1);
	pauli[2].set(1,1, -1);
	
	//Create model Hamiltonian:
	Random::seed(0);
	std::vector<LindbladFile::Kpoint> kArray(nK);
	for(LindbladFile::Kpoint& k: kArray)
	{	k.nInner = k.nOuter = nBands;
		k.innerStart = 0;
		k.E.assign(nBands, 0.);  //EY system: no spin split
		for(int i=0; i<3; i++)
		{   k.P[i] = zeroes(nBands, nBands);
			k.S[i] = (1. - 2.*spinMixSq) * pauli[i]; //in Sz diagonal basis WLOG
			k.L[i] = zeroes(nBands, nBands);
			for(int j=0; j<3; j++)
				k.L[i] += (sigmaL / sqrt(3)) * Random::normal() * pauli[j];
		}
	}

	//Add defect matrix elements (effect controlled by defectFraction in lindblad run)
	double sigmaDiag = 1.0 / sqrt(fracNZ * tauP * (4 * M_PI));
	double sigmaFlip = sigmaDiag * sqrt(spinMixSq);
	std::vector<LindbladFile::Kpoint*> kPair(2);
	for(int ik1=0; ik1<nK; ik1++)
	{	kPair[0] = &kArray[ik1];
		for(int ik2=0; ik2<ik1; ik2++)
			if(Random::uniform() < fracNZ) //sparsity
			{	kPair[1] = &kArray[ik2];

				//Create 2x2 random scattering matrix:
				matrix M = sigmaDiag * Random::normal() * eye(2);
				for(int i=0; i<3; i++)
					M += sigmaFlip * Random::normal() * pauli[i];

				//Add matrix elements to each k in pair:
				for(int iPair=0; iPair<2; iPair++)
				{	LindbladFile::GePhEntry g;
					g.jk = (iPair ? ik1 : ik2);  //partner k index
					g.omegaPh = 0.; //defect (not e-ph)
					g.G.init(nBands, nBands);
					for(int iBand=0; iBand<nBands; iBand++)
						for(int jBand=0; jBand<nBands; jBand++)
							g.G.push_back(SparseEntry{iBand, jBand, M(iBand, jBand)});
					kPair[iPair]->GePh.push_back(g);
					M = dagger(M); //set h.c. for opposite direction
				}
			}
	}

	//Prepare the file header:
	LindbladFile::Header h;
	h.dmuMin = 0;
	h.dmuMax = 0;
	h.Tmax = DBL_MAX;
	h.pumpOmegaMax = 0;
	h.probeOmegaMax = 0;
	h.nk = nK;
	h.nkTot = nK;
	h.ePhEnabled = true;
	h.spinorial = true;
	h.spinWeight = 1;
	h.R = matrix3<>(1, 1, 1);
	h.haveL = true;

	//Compute offsets to each k-point within file:
	std::vector<size_t> byteOffsets(h.nk);
	byteOffsets[0] = h.nBytes() + h.nk*sizeof(size_t); //offset to first k-point (header + byteOffsets array)
	for(size_t ik=0; ik+1<h.nk; ik++)
		byteOffsets[ik+1] = byteOffsets[ik] + kArray[ik].nBytes(h);

	//Write file:
	if(mpiWorld->isHead())
	{	FILE* fp = fopen("ldbd.dat", "w");
		// --- header
		std::ostringstream oss;
		h.write(oss);
		fwrite(oss.str().data(), 1, h.nBytes(), fp);
		// --- byte offsets
		fwrite(byteOffsets.data(), sizeof(size_t), byteOffsets.size(), fp);
		// --- data for each k-point
		for(const LindbladFile::Kpoint& k: kArray)
		{	oss.str(std::string());
			k.write(oss, h);
			fwrite(oss.str().data(), 1, k.nBytes(h), fp);
		}
		fclose(fp);
	}
	FeynWann::finalize();
	return 0;
}
