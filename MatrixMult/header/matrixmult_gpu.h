#ifndef _MATRIXMULT_GPU_H
#define _MATRIXMULT_GPU_H

#define TILE_WIDTH 16

//MatrixMult standard version GPU (parallel)
__global__ void cudaMatrixMult(float *d_M, float *d_N, double *d_P, int width);
//MatrixMult GPU with shared Memory
__global__ void cudaMatrixMultWithSMem(float *d_M, float *d_N, double *d_P, int width);

#endif /* _MATRIXMULT_GPU_H */
