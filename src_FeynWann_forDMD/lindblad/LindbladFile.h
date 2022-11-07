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

#ifndef FEYNWANN_LINDBLADFILE_H
#define FEYNWANN_LINDBLADFILE_H

#include <core/MPIUtil.h>
#include <SparseMatrix.h>
#include <FeynWann.h>

//! Data type for lindblad dynamics integration
typedef std::vector<double> DM1;

//---- Functions required by Integrator on DM1 ----
inline void axpy(double a, const DM1& x, DM1& y)
{	assert(x.size()==y.size());
	eblas_daxpy(x.size(), a, x.data(),1, y.data(),1);
}
inline DM1& operator*=(DM1& x, double s)
{	eblas_dscal(x.size(), s, x.data(),1);
	return x;
}
inline double dot(const DM1& A, const DM1& B)
{	double result = eblas_ddot(A.size(), A.data(),1, B.data(),1);
	mpiWorld->allReduce(result, MPIUtil::ReduceSum, true);
	return result;
}
inline DM1 clone(const DM1& x) { return DM1(x); }


//! Structures stored into sparse lindblad files
namespace LindbladFile
{
	static const size_t markerLen = 4; //Length of section markers in file
	
	//! Global file header
	struct Header
	{	static constexpr const char* marker = "LDBD";
		double dmuMin, dmuMax, Tmax, pumpOmegaMax, probeOmegaMax; //mu, T and pump/probe frequency range accounted for
		size_t nk, nkTot; //number of selected k-points and original total k-points (1/nkTot is BZ integration weight)
		bool ePhEnabled, spinorial; //whether e-ph and spinorial info are available
		int spinWeight; //spin factor in BZ integration
		matrix3<> R; //unit cell lattice vectors
		bool haveL; //whether angular momentum is included in data
		
		size_t nBytes() const
		{	return sizeof(char)*markerLen + sizeof(double)*5 + sizeof(size_t)*2 + sizeof(bool)*2 + sizeof(int) + sizeof(matrix3<>);
		}
		void write(std::ostream& out) const
		{	out.write((const char*)marker, sizeof(char)*markerLen);
			out.write((const char*)&dmuMin, sizeof(double)*5);
			out.write((const char*)&nk, sizeof(size_t)*2);
			out.write((const char*)&ePhEnabled, sizeof(bool)*2);
			int spinWeightFlags = encodeFlags();
			out.write((const char*)&spinWeightFlags, sizeof(int));
			out.write((const char*)&R, sizeof(matrix3<>));
		}
		void read(MPIUtil::File fp, const MPIUtil* mpiUtil)
		{	//Read and check marker:
			char markerIn[markerLen];
			mpiUtil->fread(markerIn, sizeof(char), markerLen, fp);
			if(strncmp(markerIn, marker, markerLen) != 0)
			{	fprintf(stderr, "File format error: could not find LDBD header.\n");
				mpiUtil->exit(1);
			}
			//Read data:
			mpiUtil->fread(&dmuMin, sizeof(double), 5, fp);
			mpiUtil->fread(&nk, sizeof(size_t), 2, fp);
			mpiUtil->fread(&ePhEnabled, sizeof(bool), 2, fp);
			int spinWeightFlags; mpiUtil->fread(&spinWeightFlags, sizeof(int), 1, fp);
			decodeFlags(spinWeightFlags);
			mpiUtil->fread(&R, sizeof(matrix3<>), 1, fp);
		}
		
		//Encode flags with spinWeight for backward compatibility with old data files:
		int encodeFlags() const
		{	int result = spinWeight; //always 1 or 2
			if(haveL) result |= 4;
			return result;
		}
		void decodeFlags(int spinWeightFlags)
		{	spinWeight = spinWeightFlags & 3; //lowest 2 bits contain spinWeight (1 or 2)
			haveL = (spinWeightFlags & 4); //third bit encodes L flag
		}
	};
	
	//! E-ph coupling to a specific k and for a phonon mode
	struct GePhEntry
	{	static constexpr const char* marker = "GEPH";
		size_t jk; //index of second k-point
		double omegaPh; //phonon frequency
		SparseMatrix G; //e-ph matrix elements (with energy conservation sqrt included). Note that nRows, nCols are not stored/retrieved from file.
		SparseMatrix Am, Ap; //G with sqrt(nPh) and sqrt(nPh+1) respectively multiplied (not stored by lindbladInit, but computed on startup in lindblad) 
		
		size_t nBytes() const
		{	return sizeof(char)*markerLen + sizeof(size_t) + sizeof(double)
				+ sizeof(size_t)+sizeof(SparseEntry)*G.size(); //storage for G.size() and then its entries
		}
		void write(std::ostream& out) const
		{	out.write((const char*)marker, sizeof(char)*markerLen);
			out.write((const char*)&jk, sizeof(size_t));
			out.write((const char*)&omegaPh, sizeof(double));
			size_t Gsize = G.size();
			out.write((const char*)&Gsize, sizeof(size_t));
			out.write((const char*)G.data(), sizeof(SparseEntry)*Gsize);
		}
		void read(MPIUtil::File fp, const MPIUtil* mpiUtil)
		{	//Read and check marker:
			char markerIn[markerLen];
			mpiUtil->fread(markerIn, sizeof(char), markerLen, fp);
			if(strncmp(markerIn, marker, markerLen) != 0)
			{	fprintf(stderr, "File format error: could not find GEPH header.\n");
				mpiUtil->exit(1);
			}
			//Read data:
			mpiUtil->fread(&jk, sizeof(size_t), 1, fp);
			mpiUtil->fread(&omegaPh, sizeof(double), 1, fp);
			size_t Gsize;
			mpiUtil->fread(&Gsize, sizeof(size_t), 1, fp);
			G.resize(Gsize);
			mpiUtil->freadData(G, fp);
		}
		void send(MPIUtil* mpiUtil, int dest, int tag) const
		{	mpiUtil->send(jk, dest, tag);
			mpiUtil->send(omegaPh, dest, tag);
			size_t Gsize = G.size();
			mpiUtil->send(Gsize, dest, tag);
			mpiUtil->send((const char*)G.data(), sizeof(SparseEntry)*Gsize, dest, tag);
		}
		void recv(MPIUtil* mpiUtil, int src, int tag)
		{	mpiUtil->recv(jk, src, tag);
			mpiUtil->recv(omegaPh, src, tag);
			size_t Gsize;
			mpiUtil->recv(Gsize, src, tag);
			G.resize(Gsize);
			mpiUtil->recv((char*)G.data(), sizeof(SparseEntry)*Gsize, src, tag);
		}
		
		//For sorting partner lists (by partner index and then phonon frequency):
		inline bool operator<(const GePhEntry& other) const
		{	if(jk != other.jk) return jk < other.jk;
			else return omegaPh < other.omegaPh;
		}
		
		//For searching partner lists:
		inline bool operator<(const size_t jk2) const
		{	return jk < jk2;
		}
		
		//Initialize Am and Ap given T:
		void initA(double T, double defectFraction)
		{	Am.clear(); Am.init(G.nRows(), G.nCols(), G.size());
			Ap.clear(); Ap.init(G.nRows(), G.nCols(), G.size());
			double mWeight = 0., pWeight = 0.; //weights for Am and Ap elements
			if(omegaPh)
			{	//Phonon case:
				double nPh = (T>0 and omegaPh>1e-3*T) ? bose(omegaPh/T) : 0.;
				mWeight = sqrt(nPh);
				pWeight = sqrt(nPh+1);
			}
			else
			{	//Defect case
				mWeight = sqrt(defectFraction);
				pWeight = sqrt(defectFraction);
			}
			for(const SparseEntry& se: G)
			{	SparseEntry sm = se; sm.val *= mWeight; Am.push_back(sm);
				SparseEntry sp = se; sp.val *= pWeight; Ap.push_back(sp);
			}
		}
	};
	
	//! K-point header
	struct Kpoint
	{	static constexpr const char* marker = "\nKPT";
		vector3<> k; //k-point in reciprocal lattice coordinates
		int nInner; //number of bands in the inner pump-active window
		int nOuter; //number of bands in the outer probe-active window
		int innerStart; //start of inner window relative to outer window
		
		diagMatrix E; //energies (dim: nOuter)
		matrix P[3]; //momentum matrix elements (dim: nInner x nOuter each)
		matrix S[3]; //spin matrix elements (dim: nInner x nInner each, only if spinorial)
		matrix L[3]; //orbital anguler momentum matrix elements (dim: nInner x nInner each, only if haveL)
		std::vector<GePhEntry> GePh; //e-ph matrix elements (only if ePhEnabled)
		
		size_t nBytes(const Header& h) const
		{	size_t dataSize = sizeof(char)*markerLen + sizeof(vector3<>) + sizeof(int)*3
				+ nOuter*sizeof(double) //E
				+ 3*nInner*nOuter*sizeof(complex); //P
			if(h.spinorial)
				dataSize += 3*nInner*nInner*sizeof(complex); //S
			if(h.haveL)
				dataSize += 3*nInner*nInner*sizeof(complex); //L
			if(h.ePhEnabled)
			{	dataSize += sizeof(size_t); //to store number of g
				for(const GePhEntry& g: GePh)
					dataSize += g.nBytes();
			}
			return dataSize;
		}
		void write(std::ostream& out, const Header& h) const
		{	out.write((const char*)marker, sizeof(char)*markerLen);
			out.write((const char*)&k, sizeof(vector3<>));
			out.write((const char*)&nInner, sizeof(int)*3);
			out.write((const char*)E.data(), sizeof(double)*E.size());
			for(int iDir=0; iDir<3; iDir++)
				out.write((const char*)P[iDir].data(), sizeof(complex)*P[iDir].nData());
			if(h.spinorial)
			{	for(int iDir=0; iDir<3; iDir++)
					out.write((const char*)S[iDir].data(), sizeof(complex)*S[iDir].nData());
			}
			if(h.haveL)
			{	for(int iDir=0; iDir<3; iDir++)
					out.write((const char*)L[iDir].data(), sizeof(complex)*L[iDir].nData());
			}
			if(h.ePhEnabled)
			{	size_t Gsize = GePh.size();
				out.write((const char*)&Gsize, sizeof(size_t));
				for(const GePhEntry& g: GePh)
					g.write(out);
			}
		}
		void read(MPIUtil::File fp, const MPIUtil* mpiUtil, const Header& h)
		{	//Read and check marker:
			char markerIn[markerLen];
			mpiUtil->fread(markerIn, sizeof(char), markerLen, fp);
			if(strncmp(markerIn, marker, markerLen) != 0)
			{	fprintf(stderr, "File format error: could not find KPT header (%s).\n", markerIn);
				mpiUtil->exit(1);
			}
			//Read data:
			mpiUtil->fread(&k, sizeof(vector3<>), 1, fp);
			mpiUtil->fread(&nInner, sizeof(int), 3, fp);
			E.resize(nOuter);
			mpiUtil->freadData(E, fp);
			for(int iDir=0; iDir<3; iDir++)
			{	P[iDir] = zeroes(nInner, nOuter);
				mpiUtil->freadData(P[iDir], fp);
			}
			if(h.spinorial)
			{	for(int iDir=0; iDir<3; iDir++)
				{	S[iDir] = zeroes(nInner, nInner);
					mpiUtil->freadData(S[iDir], fp);
				}
			}
			if(h.haveL)
			{	for(int iDir=0; iDir<3; iDir++)
				{	L[iDir] = zeroes(nInner, nInner);
					mpiUtil->freadData(L[iDir], fp);
				}
			}
			if(h.ePhEnabled)
			{	size_t Gsize;
				mpiUtil->fread(&Gsize, sizeof(size_t), 1, fp);
				GePh.resize(Gsize);
				for(GePhEntry& g: GePh)
					g.read(fp, mpiUtil);
			}
		}
	};
}

#endif //FEYNWANN_LINDBLADFILE_H
