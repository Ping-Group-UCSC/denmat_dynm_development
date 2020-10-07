#pragma once
#include <mymp.h>
#include <Random.h>
#include <myarray.h>
#include <sparse_matrix.h>
using namespace std;

struct sparse2D{
public:
	// convert dense 2D array A[k][{i,j}]
	// to a sparse form storing all non-zero(not tiny) elements in a continuous 2D list
	// with indexing information storing other help 1D arrays
	// Shankar's way could be another option where A is converted to std::vector<SparseMatrix> and SparseMatrix stores both data and indexing

	mymp *mp; // parallel
	size_t nk; // nk is size of 1st dimension of A
	// the second dimension of 2D array A[k][ij] often stores a matrix
	// ij is the combination of two band indeces {i,j}
	int ni, nj, nij;
	// threshold for sparsifying
	double thrsh;
	// ns_tot is total number of elements of 2D sparse array
	size_t ns_tot, nk_glob, ns_tot_glob;
	// S stores data, i and j store indeces
	complex **S, *Spool;
	int **i, *ipool, **j, *jpool;
	// ns[ik] - number of sparse elements of ik; is0[ik] - position of starting element of S[ik][0] in 2D list S
	int *ns, *is0;
	// smat[ik] will point to a struct variable storing S[ik], i[ik] and j[ik]
	sparse_mat **smat;

	sparse2D(complex **A, size_t nk, int ni, int nj, double thrsh = 1e-40)
		: thrsh(thrsh), nk(nk), ni(ni), nj(nj), nij(ni*nj), ns_tot(0), mp(nullptr)
	{
		get_ns_tot(A, 1e-10);
		get_ns_tot(A, 1e-20);
		get_ns_tot(A, 1e-30);
		get_ns_tot(A, thrsh);
	}
	sparse2D(mymp *mp, complex **A, size_t nk, int ni, int nj, double thrsh = 1e-40)
		: thrsh(thrsh), nk(nk), ni(ni), nj(nj), nij(ni*nj), ns_tot(0), mp(mp)
	{
		nk_glob = nk;
		mp->allreduce(nk_glob);

		get_ns_tot(A, 1e-10);
		get_ns_tot(A, 1e-20);
		get_ns_tot(A, 1e-30);
		get_ns_tot(A, thrsh);
	}
	~sparse2D(){
		delete[] S[0]; delete i[0]; delete j[0];
		delete[] Spool; delete[] ipool; delete[] jpool;
		delete[] S; delete[] i; delete[] j;
		delete[] ns;
	}

	void get_ns_tot(complex **A, double thr){
		for (size_t ik = 0; ik < nk; ik++)
			get_ns_tot(ik, A[ik], thr);
		if (mp){
			ns_tot_glob = ns_tot;
			mp->allreduce(ns_tot_glob);
			if (mp->ionode) printf("\nthr= %10.3le ns_tot_glob= %lu (%lu = %lu*%d)\n", thr, ns_tot_glob, nk_glob*nij, nk_glob, nij);
		}
		else if (mpkpair.inited()){
			if (mpkpair.ionode) printf("\nthr= %10.3le ns_tot= %lu (%lu = %lu*%d) (for ionode)\n", thr, ns_tot, nk*nij, nk, nij);
		}
	}
	void get_ns_tot(size_t ik, complex* A, double thr){
		if (ik == 0){
			ns_tot = 0;
			if (!ns) delete[] ns;
			if (!is0) delete[] is0;
			ns = new int[nk]; is0 = new int[nk];
		}
		is0[ik] = ns_tot;
		ns[ik] = 0;
		for (int ij = 0; ij < nij; ij++)
		if (abs(A[ij]) > thr)
			ns[ik]++;
		ns_tot += ns[ik];
	}

	void sparse(complex** A, bool do_test = false){
		for (size_t ik = 0; ik < nk; ik++)
			sparse(ik, A[ik]);
		if (do_test) zgemm_test(A);
		dealloc_array(A);
	}

	void sparse(size_t ik, complex* A){
		if (ik == 0){
			S = new complex*[nk];
			this->i = new int*[nk];
			this->j = new int*[nk];
			Spool = new complex[ns_tot]{c0};
			ipool = new int[ns_tot]();
			jpool = new int[ns_tot]();
			smat = new sparse_mat*[nk];
		}

		S[ik] = Spool; i[ik] = ipool; j[ik] = jpool;
		smat[ik] = new sparse_mat();
		smat[ik]->s = Spool; smat[ik]->i = ipool; smat[ik]->j = jpool;
		int nsk = 0;
		for (int i = 0; i < ni; i++)
		for (int j = 0; j < nj; j++)
			if (abs(A[i*ni + j]) > thrsh){
				S[ik][nsk] = A[i*ni + j];
				this->i[ik][nsk] = i;
				this->j[ik][nsk] = j;
				nsk++;
			}
		Spool += nsk; ipool += nsk; jpool += nsk;
		smat[ik]->ns = nsk;
	}

	void zgemm_test(complex **A){
		Random::seed(nk);
		size_t ik = (size_t)Random::uniformInt(nk);
		complex *densLeft = new complex[ni]; // matrix 1*ni
		complex *densRight = new complex[nj]; // matrix nj*1
		random_array(densLeft, ni);
		random_array(densRight, nj);
		complex *cdensleft = new complex[ni];
		complex *cdensRight = new complex[ni];
		complex *csparseleft = new complex[nj];
		complex *csparseRight = new complex[nj];

		zgemm_interface(cdensleft, A[ik], densRight, ni, 1, nj);
		zgemm_interface(cdensRight, densLeft, A[ik], 1, nj, ni);
		sparse_zgemm(csparseleft, true, smat[ik]->s, smat[ik]->i, smat[ik]->j, smat[ik]->ns, densRight, ni, 1, nj);
		sparse_zgemm(csparseRight, false, smat[ik]->s, smat[ik]->i, smat[ik]->j, smat[ik]->ns, densLeft, 1, nj, ni);

		if (mp){
			for (int ip = 0; ip < mp->nprocs; ip++){
				if (ip == mp->myrank){
					FILE *fp = fopen("sparse2D_zgemm_test.out", "a");
					fprintf(fp, "\nrank= %d ik= %lu\n", mp->myrank, ik);
					fprintf_complex_mat(fp, A[ik], ni, nj, "A[ik]:");
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
				printf_complex_mat(A[ik], ni, nj, "A[ik]:");
				printf_complex_mat(S[ik], 1, ns[ik], "S[ik]:");

				printf_complex_mat(cdensleft, 1, ni, "\ncdensleft:");
				printf_complex_mat(csparseleft, 1, ni, "csparseleft:");
				printf_complex_mat(cdensRight, 1, nj, "cdensRight:");
				printf_complex_mat(csparseRight, 1, nj, "csparseRight:");
			}
	}
};