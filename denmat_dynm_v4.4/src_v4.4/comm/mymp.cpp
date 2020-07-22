#include "mymp.h"

mymp mpkpair;
mymp mpkpair2;
mymp mpk;

bool mymp::distribute_var(string routine, size_t nvar){
	MPI_Barrier(MPI_COMM_WORLD);
	for (int iProc = 0; iProc < nprocs; iProc++)
		endArr.push_back((nvar * (iProc + 1)) / nprocs);
	varstart = start(myrank);
	varend = end(myrank);
	/*
	int message = 999;
	if (myrank == 0) {
		MPI_Send(&message, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
		printf("myrank = %d, varstart = %lu, varend = %lu\n", myrank, varstart, varend);
	}
	else {
		int buffer;
		MPI_Status status;
		MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
		MPI_Get_count(&status, MPI_INT, &buffer);
		if (buffer == 1) {
			printf("myrank = %d, varstart = %lu, varend = %lu\n", myrank, varstart, varend);
			MPI_Recv(&message, buffer, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
			if (myrank + 1 != nprocs) {
				MPI_Send(&message, 1, MPI_INT, myrank+1, 0, MPI_COMM_WORLD);
			}
		};
	};
	*/
	MPI_Barrier(MPI_COMM_WORLD);
    return true;
}

void mymp::allreduce(size_t& m, MPI_Op op){
	MPI_Allreduce(MPI_IN_PLACE, &m, 1, my_MPI_SIZE_T, op, MPI_COMM_WORLD);
}

void mymp::allreduce(double& m, MPI_Op op){
	MPI_Allreduce(MPI_IN_PLACE, &m, 1, MPI::DOUBLE, op, MPI_COMM_WORLD);
}

void mymp::allreduce(complex **m, int n1, int n2, MPI_Op op){
	MPI_Barrier(MPI_COMM_WORLD);
	for (int i = 0; i < n1; i++)
		MPI_Allreduce(MPI_IN_PLACE, &m[i][0], n2, MPI::DOUBLE_COMPLEX, op, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
}
void mymp::allreduce(double **m, int n1, int n2, MPI_Op op){
	MPI_Barrier(MPI_COMM_WORLD);
	for (int i = 0; i < n1; i++)
		MPI_Allreduce(MPI_IN_PLACE, &m[i][0], n2, MPI::DOUBLE, op, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
}

void mymp::collect(int comm, int nprocs_lv, int varstart, int nvar, int *disp_proc, int *nvar_proc){
}
