#include <core/matrix.h>
#include <core/Random.h>
#include <core/Units.h>
#include <InputMap.h>
#include <lindblad/LindbladFile.h>


int main(int argc, char** argv)
{
	InitParams ip = FeynWann::initialize(argc, argv, "Create a (2-band) spin model system");

	//Get input parameters:
	InputMap inputMap(ip.inputFilename);
	const int nK = int(inputMap.get("nK")); //number of k-points
	const vector3<> sigmaB = inputMap.getVector("sigmaB") * Tesla; //magnitude of internal magnetic field fluctuations per direction
	const double scatterB = inputMap.get("scatterB") * Tesla; //magnitude of scattering terms written as a magnetic field
	const int nBands = 2;
	
	//Print back input parameters (converted):
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nK = %d\n", nK);
	logPrintf("sigmaB = "); sigmaB.print(globalLog, " %lg ");
	logPrintf("scatterB = %lg\n", scatterB);
	logPrintf("\n");
	
	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		FeynWann::finalize();
		return 0;
	}
	logPrintf("\n");
	
	//Create Pauli matrices:
	vector3<matrix> S;
	// Sx
	S[0] = zeroes(nBands, nBands);
	S[0].set(0,1, 1);
	S[0].set(1,0, 1);
	// Sy
	S[1] = zeroes(nBands, nBands);
	S[1].set(0,1, complex(0,-1));
	S[1].set(1,0, complex(0, 1));
	// Sz
	S[2] = zeroes(nBands, nBands);
	S[2].set(0,0,  1);
	S[2].set(1,1, -1);
	
	//Create model Hamiltonian:
	Random::seed(0);
	std::vector<LindbladFile::Kpoint> kArray(nK);
	for(LindbladFile::Kpoint& k: kArray)
	{	k.nInner = k.nOuter = nBands;
		k.innerStart = 0;

		//Create a random magnetic field hamiltonian:
		matrix H0 = zeroes(nBands, nBands);
		for(int iDir=0; iDir<3; iDir++)
			H0 += (sigmaB[iDir] * Random::normal()) * S[iDir];
		
		//Diagonalize:
		matrix V;
		H0.diagonalize(V, k.E);
		for (int i=0; i < 3; i++)
		{   k.S[i] = dagger(V) * S[i] * V;
			k.P[i] = zeroes(nBands, nBands);
		}
	}

	//Add defect matrix elements (effect controlled by defectFraction in lindblad run)
	double sigmaDefect = sqrt(scatterB / nK) / nBands;
	for(int ik1=0; ik1<nK; ik1++)
	{	LindbladFile::Kpoint& k1 = kArray[ik1];
		for(int ik2=0; ik2<ik1; ik2++)
		{	LindbladFile::Kpoint& k2 = kArray[ik2];
			//Create matrix element set connecting these k:
			LindbladFile::GePhEntry M12, M21;
			M12.jk = ik2;
			M21.jk = ik1;
			M12.omegaPh = 0.; //defect (not e-ph)
			M21.omegaPh = 0.; //defect (not e-ph)
			M12.G.init(nBands, nBands);
			M21.G.init(nBands, nBands);
			for(int b1=0; b1<nBands; b1++)
				for(int b2=0; b2<nBands; b2++)
				{	complex M = sigmaDefect * Random::normalComplex();
					M12.G.push_back(SparseEntry{b1, b2, M});
					M21.G.push_back(SparseEntry{b2, b1, M.conj()});
				}
			//Add to kpoints:
			k1.GePh.push_back(M12);
			k2.GePh.push_back(M21);
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
	h.haveL = false;

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
