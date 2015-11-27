#include "../header/transposeCopyMatrixGPU.h"

/*
  matrix copy kernel
*/
__global__ void copyMatrix(float* outMatrix, float* inMatrix, int width, int height, int nreps){
  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
  int memIndex = xIndex + width * yIndex;
  //measureloop
  for(int measure = 0; measure < nreps; ++measure){
    //workloop
    for(int i = 0; i < TILE_DIM; i+=BLOCK_ROWS){
      outMatrix[memIndex + i * width] = inMatrix[memIndex + i * width];
    }
    __syncthreads();
  }
}
