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
    //workloop
  }
  //measureloop
}

//parallel copy kernel invocation
__host__ void measureCopyKernel(float* inputMatrix, float* ctrCMatrix, dim3 gridDim, dim3 blockDim, int matrixWidth, int matrixHeight, int nReps){
  int memSizeMatrix = 0;
  int sizeMatrix = 0;
  double tStart = 0;
  double tStop = 0;
  float *d_inCopyMatrix; //input Matrix for parallel gpu copy kernel
  float *d_outCopyMatrix; //output Matrix for parallel gpu copy kernel
  float *h_outputCopyMatrix;
  preProcess("gpu parallel copyMatrix");

  sizeMatrix = matrixWidth * matrixHeight;
  memSizeMatrix = sizeMatrix * sizeof(float);

  //device mem alloc
  cudaErr(cudaMalloc((void**)&d_inCopyMatrix, memSizeMatrix));//MemSize anders bestimmen
  cudaErr(cudaMalloc((void**)&d_outCopyMatrix, memSizeMatrix));
  //Daten zum devices Kopieren!
  //InMatrix zum devices kopieren
  cudaErr(cudaMemcpy(d_inCopyMatrix, inputMatrix, memSizeMatrix, cudaMemcpyHostToDevice));
  cudaErr(cudaDeviceSynchronize());

  //Warmup round
  copyMatrix<<<gridDim,blockDim>>>(d_outCopyMatrix, d_inCopyMatrix, matrixWidth, matrixHeight, 1);

  //Loop inside
  tStart = omp_get_wtime();
  copyMatrix<<<gridDim,blockDim>>>(d_outCopyMatrix, d_inCopyMatrix, matrixWidth, matrixHeight, nReps);
  cudaErr(cudaDeviceSynchronize());
  tStop = omp_get_wtime();
  postProcess(nReps, sizeMatrix, (tStop-tStart), "inner loop");

  //Loop outside
  tStart = omp_get_wtime();
  for(int i = 0; i< nReps; ++i){
    copyMatrix<<<gridDim,blockDim>>>(d_outCopyMatrix, d_inCopyMatrix, matrixWidth, matrixHeight, 1);
    cudaErr(cudaDeviceSynchronize());
  }
  tStop = omp_get_wtime();
  postProcess(nReps, sizeMatrix, (tStop-tStart), "outer loop");

  //daten vom devices kopieren!!!
  h_outputCopyMatrix = (float*)malloc(memSizeMatrix);
  cudaErr(cudaMemcpy(h_outputCopyMatrix, d_outCopyMatrix, memSizeMatrix, cudaMemcpyDeviceToHost));
  //compare Matrix
  compareMatrix(ctrCMatrix, h_outputCopyMatrix, sizeMatrix, "CPU CopyMatrix","GPU CopyMatrix" );

  //DevicesSpeicher freigeben
  cudaErr(cudaFree(d_inCopyMatrix));
  cudaErr(cudaFree(d_outCopyMatrix));
  //Hostspeicher freigeben
  free(h_outputCopyMatrix);
}
