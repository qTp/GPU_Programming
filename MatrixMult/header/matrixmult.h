#ifndef _MATRIXMULT_H
#define _MATRIXMULT_H

//MatrixMult standard version GPU (parallel)
__global__ void cudaMatrixMult(float *d_M, float *d_N, double *d_P, int width);
//MatrixMult GPU with shared Memory
__global__ void cudaMatrixMultWithSMem(float *d_M, float *d_N, double *d_P, int width);

//Matrix Mult serial CPU version
void serialMatrixMult(float *M, float *N, double *P, int width);


#endif /* _MATRIXMULT_H */
