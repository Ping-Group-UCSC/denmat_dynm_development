#pragma once
#include <mymp.h>
#include <myio.h>
#include <Random.h>
#include <myarray.h>
#include <sparse_matrix.h>
using namespace std;

struct sparse2D{
public:
	// convert dense 2D array A[k][{i,j}]
	// to a sparse form storing all non-zero(not tiny) elements in an array of sparse_mat - smat[k]. See sparse_mat in sparse_mat.h

	mymp *mp; // parallel
	size_t nk, ns_tot, nk_glob, ns_tot_glob; // nk is size of 1st dimension of A, ns_tot is total number of elements of 2D sparse array
	int ni, nj, nij; // the second dimension of 2D array A[k][ij] often stores a matrix. ij is the combination of two band indeces {i,j}
	double thrsh; // threshold for sparsifying
	sparse_mat **smat;
	string fnamens, fnames, fnamei, fnamej;
	FILE *fpns, *fps, *fpi, *fpj;
	size_t is0; // starting position of s, i and j of smat at the corresponding files in mpi mode

	// note that when smat will be updated, in principle, ns_tot, ns_tot_glob, is0 should all be updated
	// but given that 

	sparse2D(complex **A, size_t nk, int ni, int nj, double thrsh = 1e-40)
		: thrsh(thrsh), nk(nk), ni(ni), nj(nj), nij(ni*nj), ns_tot(0), is0(is0), mp(nullptr)
	{
		smat = new sparse_mat*[nk];
		if (A != nullptr){
			get_ns_tot(A, 1e-20);
			get_ns_tot(A, thrsh);
		}
		else alloc_empty();
	}
	sparse2D(mymp *mp, complex **A, int ni, int nj, double thrsh = 1e-40)
		: mp(mp), thrsh(thrsh), nk(mp->varend - mp->varstart), ni(ni), nj(nj), nij(ni*nj), ns_tot(0), is0(is0)
	{
		smat = new sparse_mat*[nk];
		nk_glob = nk;
		mp->allreduce(nk_glob);
		if (A != nullptr){
			get_ns_tot(A, 1e-20);
			get_ns_tot(A, thrsh);
		}
		else alloc_empty();
	}
	void alloc_empty(){
		for (size_t ik = 0; ik < nk; ik++)
			smat[ik] = new sparse_mat();
		print_ns_tot(thrsh);
	}
	sparse2D(mymp *mp, string fnamens, string fnames, string fnamei, string fnamej, int ni, int nj)
		: mp(mp), fnamens(fnamens), fnames(fnames), fnamei(fnamei), fnamej(fnamej), nk(mp->varend - mp->varstart), ni(ni), nj(nj), nij(ni*nj), ns_tot(0), is0(0), thrsh(1e-40)
	{
		smat = new sparse_mat*[nk];
		nk_glob = nk;
		mp->allreduce(nk_glob);

		get_ns_tot_fromfile();
		if (mp != nullptr) mp->varstart_from_nvar(is0, ns_tot);
	}
	~sparse2D(){
		if ((mp != nullptr && mp->ionode) || mp == nullptr) { printf("destroy this sparse2D object\n"); fflush(stdout); }
		for (int ik; ik < nk; ik++){ delete smat[ik]; smat[ik] = nullptr;  }
		delete[] smat; smat = nullptr;
	}

	void get_ns_tot(complex **A, double thr){
		for (size_t ik = 0; ik < nk; ik++)
			get_ns_tot(ik, A[ik], thr);
		print_ns_tot(thr);
	}
	void get_ns_tot(size_t ik, complex* A, double thr){
		if (ik == 0) ns_tot = 0;
		for (int ij = 0; ij < nij; ij++)
			if (abs(A[ij]) > thr) ns_tot++;
	}
	void get_ns_tot_fromfile(){
		for (size_t ik = 0; ik < nk; ik++)
			get_nsk_fromfile(ik);
		print_ns_tot();
	}
	void get_nsk_fromfile(size_t ik){
		if (ik == 0){
			ns_tot = 0;
			fpns = fopen(fnamens.c_str(), "rb");
			if (mp != nullptr) fseek(fpns, mp->varstart * sizeof(int), SEEK_SET);
		}
		int nsk = 0;
		fread(&nsk, sizeof(int), 1, fpns);
		ns_tot += nsk;
		if (ik == nk - 1) fclose(fpns);
	}
	void print_ns_tot(double thr = -1){
		if (mp != nullptr){ ns_tot_glob = ns_tot; mp->allreduce(ns_tot_glob); }
		if (mp != nullptr) { if (mp->ionode) printf("\nthr= %10.3le ns_tot_glob= %lu (%lu = %lu*%d)\n", thr, ns_tot_glob, nk_glob*nij, nk_glob, nij); }
		else if (mpkpair.inited() && mpkpair.ionode) printf("\nthr= %10.3le ns_tot= %lu (%lu = %lu*%d) (for ionode)\n", thr, ns_tot, nk*nij, nk, nij);
		else printf("\nthr= %10.3le ns_tot= %lu (%lu = %lu*%d)\n", thr, ns_tot, nk*nij, nk, nij);
	}

	void sparse(complex** A, bool do_test = false){
		if ((mp != nullptr && mp->ionode) || mp == nullptr) printf("start sparsing matrix vector\n");
		for (size_t ik = 0; ik < nk; ik++)
			smat[ik] = new sparse_mat(A[ik], ni, nj, thrsh);
		if (do_test) zgemm_test();
		dealloc_array(A);
	}

	void read_smat(bool do_test = false){
		for (size_t ik = 0; ik < nk; ik++)
			read_smat(ik);
		if (do_test) zgemm_test();
	}
	void read_smat(size_t ik){
		if (ik == 0){
			fpns = fopen(fnamens.c_str(), "rb");
			fps = fopen(fnames.c_str(), "rb");
			fpi = fopen(fnamei.c_str(), "rb");
			fpj = fopen(fnamej.c_str(), "rb");

			if (mp != nullptr){
				fseek(fpns, mp->varstart * sizeof(int), SEEK_SET);
				fseek(fps, is0 * 2 * sizeof(double), SEEK_SET);
				fseek(fpi, is0 * sizeof(int), SEEK_SET);
				fseek(fpj, is0 * sizeof(int), SEEK_SET);
			}
		}
		smat[ik] = new sparse_mat(fpns, fps, fpi, fpj);
		if (ik == nk - 1) { fclose(fpns); fclose(fps); fclose(fpi); fclose(fpj); }
	}

	void write_smat(){
		write_smat(this->fnamens, this->fnames, this->fnamei, this->fnamej);
	}
	void write_smat(string fnamens, string fnames, string fnamei, string fnamej){
		if (exists(fnames)) return; // avoid overwriting
		for (int i = 0; i < mp->nprocs; i++){
			if (i == mp->myrank){
				fpns = fopen(fnamens.c_str(), "ab");
				fps = fopen(fnames.c_str(), "ab");
				fpi = fopen(fnamei.c_str(), "ab");
				fpj = fopen(fnamej.c_str(), "ab");

				for (int ik = 0; ik < nk; ik++)
					smat[ik]->write_to_files(fpns, fps, fpi, fpj);
				fclose(fpns); fclose(fps); fclose(fpi); fclose(fpj);
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}
	}

	void zgemm_test(){
		Random::seed(nk);
		size_t ik = (size_t)Random::uniformInt(nk);
		complex *A = smat[ik]->todense(ni, nj);
		complex *densLeft = new complex[ni]; // matrix 1*ni
		complex *densRight = new complex[nj]; // matrix nj*1
		random_array(densLeft, ni);
		random_array(densRight, nj);
		complex *cdensleft = new complex[ni];
		complex *cdensRight = new complex[ni];
		complex *csparseleft = new complex[nj];
		complex *csparseRight = new complex[nj];

		zgemm_interface(cdensleft, A, densRight, ni, 1, nj);
		zgemm_interface(cdensRight, densLeft, A, 1, nj, ni);
		sparse_zgemm(csparseleft, true, smat[ik], densRight, ni, 1, nj);
		sparse_zgemm(csparseRight, false, smat[ik], densLeft, 1, nj, ni);

		if (mp != nullptr){
			for (int ip = 0; ip < mp->nprocs; ip++){
				if (ip == mp->myrank){
					FILE *fp = fopen("sparse2D_zgemm_test.out", "a");
					//fprintf(fp, "\nrank= %d ik= %lu\n", mp->myrank, ik);
					fprintf_complex_mat(fp, A, ni, nj, "A[ik]:");
					fprintf_complex_mat(fp, smat[ik]->s, 1, smat[ik]->ns, "S[ik]:");

					fprintf_complex_mat(fp, cdensleft, 1, ni, "cdensleft:");
					fprintf_complex_mat(fp, csparseleft, 1, ni, "csparseleft:");
					fprintf_complex_mat(fp, cdensRight, 1, nj, "cdensRight:");
					fprintf_complex_mat(fp, csparseRight, 1, nj, "csparseRight:");
					fclose(fp);
				}
				MPI_Barrier(MPI_COMM_WORLD);
			}
		}
		else if (mpkpair.inited())
			if (mpkpair.ionode){
				printf("\nik= %lu\n", ik);
				printf_complex_mat(A, ni, nj, "A[ik]:");
				printf_complex_mat(smat[ik]->s, 1, smat[ik]->ns, "S[ik]:");

				printf_complex_mat(cdensleft, 1, ni, "\ncdensleft:");
				printf_complex_mat(csparseleft, 1, ni, "csparseleft:");
				printf_complex_mat(cdensRight, 1, nj, "cdensRight:");
				printf_complex_mat(csparseRight, 1, nj, "csparseRight:");
			}
	}
};