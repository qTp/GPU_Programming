#ifndef TRANSPOSECOPYMATRIX_GPU_H_
#define TRANSPOSECOPYMATRIX_GPU_H_
//cudaTool for error handling
#include "../header/cudatool.h"

#define TILE_DIM 32
#define BLOCK_ROWS 8

//#define MAX_BLOCK_DIM dim3(TILE_DIM,BLOCK_ROWS,1);
//  simple copy kernel
__global__ void copyMatrix(float* outMatrix, float* inMatrix,int width, int height, int nreps);
// measureCopyKernel
__host__ void measureCopyKernel(float* inputMatrix, float* ctrCMatrix, dim3 gridDim, dim3 blockDim, int matrixWidth, int matrixHeight, int nReps);
#endif /* TRANSPOSECOPYMATRIX_GPU_H_ */
