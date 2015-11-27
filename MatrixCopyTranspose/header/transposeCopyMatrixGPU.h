#ifndef TRANSPOSECOPYMATRIXGPU_H_
#define TRANSPOSECOPYMATRIXGPU_H_
//cudaTool for error handling
#include "../header/cudatool.h"

#define TILE_DIM 32
#define BLOCK_ROWS 8

//#define MAX_BLOCK_DIM dim3(TILE_DIM,BLOCK_ROWS,1);


//  simple copy kernel
__global__ void copyMatrix(float* inMatrix,float* outMatrix, int width, int height, int nreps);



#endif /* TRANSPOSECOPYMATRIXGPU_H_ */
