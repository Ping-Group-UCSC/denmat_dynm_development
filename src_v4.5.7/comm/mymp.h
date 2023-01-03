#pragma once
#include <algorithm>
#include <stdint.h>
#include <limits.h>
#include <scalar.h>
#include <string>
#include <mpi.h>
using namespace std;

#if SIZE_MAX == UCHAR_MAX
#define my_MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
#define my_MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
#define my_MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
#define my_MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
#define my_MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
#error "what is happening here?"
#endif

class mymp{
public:
	bool ionode;
	int nprocs, myrank;
	size_t varstart, varend;
	std::vector<size_t> endArr; // idea from JDFTx

	mymp() : nprocs(0) {}

	bool inited() { return nprocs > 0; }

	void mpi_init(){
		int flag = 0;
		MPI_Initialized(&flag);
		if (!flag) MPI_Init(NULL, NULL);
		MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
		MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
		ionode = (myrank == 0);
	};
	void mpi_finalize(){
		MPI_Finalize();
	};
	void mpi_abort(string s = "", int errorcode = MPI_SUCCESS){
		if (ionode) {
			printf("%s", s.c_str());
			MPI_Abort(MPI_COMM_WORLD, errorcode);
		}
		else
			MPI_Barrier(MPI_COMM_WORLD);
		// further will not be excuted
		MPI_Finalize();
		if (errorcode == MPI_SUCCESS) exit(EXIT_SUCCESS);
		else exit(EXIT_FAILURE);
	}

	bool distribute_var(string routine, size_t nvar);
	inline size_t start(int iProc){ return iProc ? endArr[iProc - 1] : 0; } //!< Task number that the specified process should start on
	inline size_t end(int iProc){ return endArr[iProc]; } //!< Task number that the specified process should stop before (non-inclusive)
	int whose(size_t q){ //!< Which process number should handle this task number
		if (endArr.size()>1)
			return std::upper_bound(endArr.begin(), endArr.end(), q) - endArr.begin();
		else return 0;
	}

	void allreduce(size_t& m, MPI_Op op = MPI_SUM);
	void allreduce(double& m, MPI_Op op = MPI_SUM);
	void allreduce(complex& m, MPI_Op op = MPI_SUM);
	void allreduce(double *v, int n, MPI_Op op = MPI_SUM);
	void allreduce(complex *v, int n, MPI_Op op = MPI_SUM);
	void allreduce(complex **m, int n1, int n2, MPI_Op op = MPI_SUM);
	void allreduce(complex ***m, int n1, int n2, int n3, MPI_Op op = MPI_SUM);
	void allreduce(vector<vector<double>>& m, MPI_Op op = MPI_SUM);
	void allreduce(vector<vector<complex>>& m, MPI_Op op = MPI_SUM);
	void allreduce(double **m, int n1, int n2, MPI_Op op = MPI_SUM);
	void allreduce(double ***a, int n1, int n2, int n3, MPI_Op op = MPI_SUM);
	void collect(int, int, int, int, int*, int*);
	void varstart_from_nvar(size_t& varstart, size_t nvar);
	void bcast(size_t*, int, int root = 0);
};

extern mymp mpkpair;
extern mymp mpkpair2;
extern mymp mpk;
extern mymp mpk_morek;