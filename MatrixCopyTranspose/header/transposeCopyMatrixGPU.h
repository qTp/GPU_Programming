#ifndef TRANSPOSECOPYMATRIX_GPU_H_
#define TRANSPOSECOPYMATRIX_GPU_H_
//cudaTool for error handling
#include "../header/cudatool.h"

#define TILE_DIM 32
#define BLOCK_ROWS 8

typedef void (*_kernel_)(float*, float*, int, int, int);
extern _kernel_ func;

//  simple copy kernel
__global__ void copyMatrix(float* inMatrix,float* outMatrix,int width, int height, int nreps);
// measure Kernel
__host__ void measureKernel(float* inputMatrix, float* ctrMatrix, dim3 gridDim, dim3 blockDim, int matrixWidth, int matrixHeight, int nReps, char funcName[20], _kernel_ func);

#endif /* TRANSPOSECOPYMATRIX_GPU_H_ */
