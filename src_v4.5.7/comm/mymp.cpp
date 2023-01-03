#include "mymp.h"

mymp mpkpair;
mymp mpkpair2;
mymp mpk;
mymp mpk_morek;

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
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &m, 1, my_MPI_SIZE_T, op, MPI_COMM_WORLD);
}
void mymp::allreduce(double& m, MPI_Op op){
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &m, 1, MPI_DOUBLE, op, MPI_COMM_WORLD);
}
void mymp::allreduce(complex& m, MPI_Op op){
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &m, 1, MPI_DOUBLE_COMPLEX, op, MPI_COMM_WORLD);
}

void mymp::allreduce(double *v, int n, MPI_Op op){
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, op, MPI_COMM_WORLD);
}
void mymp::allreduce(complex *v, int n, MPI_Op op){
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE_COMPLEX, op, MPI_COMM_WORLD);
}

void mymp::allreduce(complex **m, int n1, int n2, MPI_Op op){
	MPI_Barrier(MPI_COMM_WORLD);
	for (int i = 0; i < n1; i++){
		MPI_Allreduce(MPI_IN_PLACE, &m[i][0], n2, MPI_DOUBLE_COMPLEX, op, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
	}
	MPI_Barrier(MPI_COMM_WORLD);
}
void mymp::allreduce(complex ***a, int n1, int n2, int n3, MPI_Op op){
	for (int i = 0; i < n1; i++)
		allreduce(a[i], n2, n3, op);
}
void mymp::allreduce(vector<vector<double>>& m, MPI_Op op){
	MPI_Barrier(MPI_COMM_WORLD);
	for (size_t i = 0; i < m.size(); i++){
		MPI_Allreduce(MPI_IN_PLACE, m[i].data(), m[i].size(), MPI_DOUBLE, op, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
	}
	MPI_Barrier(MPI_COMM_WORLD);
}
void mymp::allreduce(vector<vector<complex>>& m, MPI_Op op){
	MPI_Barrier(MPI_COMM_WORLD);
	for (size_t i = 0; i < m.size(); i++){
		MPI_Allreduce(MPI_IN_PLACE, m[i].data(), m[i].size(), MPI_DOUBLE_COMPLEX, op, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
	}
	MPI_Barrier(MPI_COMM_WORLD);
}
void mymp::allreduce(double **m, int n1, int n2, MPI_Op op){
	MPI_Barrier(MPI_COMM_WORLD);
	//MPI_Allreduce(MPI_IN_PLACE, m[0], n1*n2, MPI_DOUBLE, op, MPI_COMM_WORLD);
	for (int i = 0; i < n1; i++){
		MPI_Allreduce(MPI_IN_PLACE, &m[i][0], n2, MPI_DOUBLE, op, MPI_COMM_WORLD);
		//MPI_Reduce(ionode ? MPI_IN_PLACE : m[i], m[i], n2, MPI_DOUBLE, op, 0, MPI_COMM_WORLD);
		//MPI_Bcast(m[i], n2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
	}
	MPI_Barrier(MPI_COMM_WORLD);
}
void mymp::allreduce(double ***a, int n1, int n2, int n3, MPI_Op op){
	for (int i = 0; i < n1; i++)
		allreduce(a[i], n2, n3, op);
}

void mymp::collect(int comm, int nprocs_lv, int varstart, int nvar, int *disp_proc, int *nvar_proc){
}

void mymp::varstart_from_nvar(size_t& varstart, size_t nvar){
	MPI_Barrier(MPI_COMM_WORLD);
	varstart = 0;
	double sum = 0;
	for (int i = 0; i < nprocs; i++){
		if (i == myrank){
			varstart = sum;
			//printf("rank= %d v0= %lu\n", i, varstart);
			sum += nvar;
		}
		allreduce(sum, MPI_MAX);
	}
}

void mymp::bcast(size_t* a, int count, int root){
	MPI_Bcast(a, count, my_MPI_SIZE_T, root, MPI_COMM_WORLD);
}
