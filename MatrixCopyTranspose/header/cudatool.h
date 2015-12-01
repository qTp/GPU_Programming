#ifndef _CUDATOOL_H
#define _CUDATOOL_H

#include <stdio.h>
#include <stdlib.h>
//timeMeasure
#include <omp.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define cudaErr(err) cudaErrT(err, __LINE__,__FILE__)
cudaError_t cudaErrT(cudaError_t err, int line, char* file );

void initMatrix(float *ip, int size);
void compareMatrix(double *P1, double *P2, int N, char name1[25], char name2[25]);
void compareMatrix(float *P1, float *P2, int N, char name1[25], char name2[25]);
void preProcess(char *_name);
void postProcess(int reps, int memSize, double tElapsed, char *_type);

#endif /* _CUDATOOL_H */
