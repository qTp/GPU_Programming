#include "../header/transposeCopyMatrixGPU.h"

/*
  matrix copy kernel
*/
__global__ void copyMatrix(float* inMatrix, float* outMatrix, int width, int height, int nreps){
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
__host__ void measureKernel(float* inputMatrix, float* ctrMatrix, dim3 gridDim, dim3 blockDim, int matrixWidth, int matrixHeight, int nReps, char *funcName, _kernel_ func){
  int memSizeMatrix = 0;
  int sizeMatrix = 0;
  double tStart = 0;
  double tStop = 0;
  float *d_inCopyMatrix; //input Matrix for parallel gpu copy kernel
  float *d_outCopyMatrix; //output Matrix for parallel gpu copy kernel
  float *h_outputCopyMatrix;

  char processName[25] = "GPU ";
  strcat(processName, funcName);

  preProcess( processName  );
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
  func<<<gridDim,blockDim>>>(d_inCopyMatrix, d_outCopyMatrix, matrixWidth, matrixHeight, 1);

  //Loop inside
  tStart = omp_get_wtime();
  for(int i = 0; i < 1; ++i){
    func<<<gridDim,blockDim>>>(d_inCopyMatrix,d_outCopyMatrix, matrixWidth, matrixHeight, nReps);
    cudaErr(cudaDeviceSynchronize());
  }
  tStop = omp_get_wtime();
  postProcess(nReps, sizeMatrix, (tStop-tStart), "inner loop");

  //Loop outside
  tStart = omp_get_wtime();
  for(int i = 0; i< nReps; ++i){
    func<<<gridDim,blockDim>>>(d_inCopyMatrix,d_outCopyMatrix, matrixWidth, matrixHeight, 1);
    cudaErr(cudaDeviceSynchronize());
  }
  tStop = omp_get_wtime();
  postProcess(nReps, sizeMatrix, (tStop-tStart), "outer loop");

  //daten vom devices kopieren!!!
  h_outputCopyMatrix = (float*)malloc(memSizeMatrix);
  cudaErr(cudaMemcpy(h_outputCopyMatrix, d_outCopyMatrix, memSizeMatrix, cudaMemcpyDeviceToHost));

  //Matrix testen
  char name1[25] = "CPU ";
  strcat(name1,funcName);
  char name2[25] = "GPU ";
  strcat(name2,funcName);
  compareMatrix(ctrMatrix, h_outputCopyMatrix, sizeMatrix, name1, name2);

  //DevicesSpeicher freigeben
  cudaErr(cudaFree(d_inCopyMatrix));
  cudaErr(cudaFree(d_outCopyMatrix));
  //Hostspeicher freigeben
  free(h_outputCopyMatrix);
}
