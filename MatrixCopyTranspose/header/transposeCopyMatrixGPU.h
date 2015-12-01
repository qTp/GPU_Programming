#ifndef TRANSPOSECOPYMATRIX_GPU_H_
#define TRANSPOSECOPYMATRIX_GPU_H_
//cudaTool for error handling
#include "../header/cudatool.h"

#define TILE_DIM 32
#define BLOCK_ROWS 8
#define DEFAULT_STREAM 0

typedef void (*_kernel_)(float*, float*, int, int, int);
extern _kernel_ func;

//  simple copy kernel
__global__ void copyMatrix(float* outMatrix, float* inMatrix, int width, int height, int nreps);
//  simple transpose kernel (naive transpose)
__global__ void transposeMatrix(float* outMatrix, float* inMatrix, int width, int height, int nreps);
// measure Kernel with omp_get_wtime
__host__ void measureKernelOMP(float* inputMatrix, float* ctrMatrix, dim3 gridDim, dim3 blockDim, int matrixWidth, int matrixHeight, int nReps, char funcName[20], _kernel_ func);
// measure Kernel with nvidia timing event
__host__ void measureKernel(float* inputMatrix, float* ctrMatrix, dim3 gridDim, dim3 blockDim, int matrixWidth, int matrixHeight, int nReps, char funcName[20], _kernel_ func);


#endif /* TRANSPOSECOPYMATRIX_GPU_H_ */
