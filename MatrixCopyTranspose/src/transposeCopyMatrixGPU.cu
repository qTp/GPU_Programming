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
/*
  matrix transpose kernel
*/
__global__ void transposeMatrix(float* outMatrix, float* inMatrix, int width, int height, int nreps){
  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
  int inIndex = xIndex + width * yIndex;
  int outIndex = yIndex + height * xIndex;
  //measureloop
  for(int measure = 0; measure < nreps; ++measure){
    //workloop
    for(int i = 0; i < TILE_DIM; i+=BLOCK_ROWS){
      outMatrix[outIndex + i] = inMatrix[inIndex + i * width];
    }
    __syncthreads();
    //workloop
  }
  //measureloop
}
/*
  Kernel invocation and time measure with nvidia timing event
*/
__host__ void measureKernel(float* inputMatrix, float* ctrMatrix, dim3 gridDim, dim3 blockDim, int matrixWidth, int matrixHeight, int nReps, char *funcName, _kernel_ func){
  int memSizeMatrix = 0;
  int sizeMatrix = 0;
  float tElapsed=0;
  cudaEvent_t tStart;
  cudaEvent_t tStop;
  float *d_inMatrix; //input Matrix for parallel gpu copy kernel
  float *d_outMatrix; //output Matrix for parallel gpu copy kernel
  float *h_outputMatrix;

  char processName[50] = "GPU ";
  strcat(processName, funcName);
  strcat(processName, " with cudaEventRecord");

  preProcess( processName  );

  sizeMatrix = matrixWidth * matrixHeight;
  memSizeMatrix = sizeMatrix * sizeof(float);

  //cuda timing events erstellen
  cudaErr(cudaEventCreate(&tStart));
  cudaErr(cudaEventCreate(&tStop));

  //device mem alloc
  cudaErr(cudaMalloc((void**)&d_inMatrix, memSizeMatrix));//MemSize anders bestimmen
  cudaErr(cudaMalloc((void**)&d_outMatrix, memSizeMatrix));

  //InMatrix zum devices Kopieren!
  cudaErr(cudaMemcpy(d_inMatrix, inputMatrix, memSizeMatrix, cudaMemcpyHostToDevice));

  //Warmup round
  func<<<gridDim,blockDim>>>(d_outMatrix, d_inMatrix, matrixWidth, matrixHeight, 1);

  //Loop inside
  cudaErr(cudaEventRecord(tStart, DEFAULT_STREAM));
  for(int i = 0; i < 1; ++i){
    func<<<gridDim,blockDim>>>(d_outMatrix, d_inMatrix, matrixWidth, matrixHeight, nReps);
    cudaErr(cudaDeviceSynchronize());
  }
  cudaErr(cudaEventRecord(tStop, DEFAULT_STREAM));
  cudaErr(cudaEventSynchronize(tStop));
  cudaErr(cudaEventElapsedTime(&tElapsed, tStart, tStop));

  postProcess(nReps, sizeMatrix, tElapsed, "inner loop");

  //Loop outside
  cudaErr(cudaEventRecord(tStart, DEFAULT_STREAM));
  for(int i = 0; i< nReps; ++i){
    func<<<gridDim,blockDim>>>(d_outMatrix, d_inMatrix, matrixWidth, matrixHeight, 1);
    cudaErr(cudaDeviceSynchronize());
  }
  cudaErr(cudaEventRecord(tStop, DEFAULT_STREAM));
  cudaErr(cudaEventSynchronize(tStop));
  cudaErr(cudaEventElapsedTime(&tElapsed, tStart, tStop));
  postProcess(nReps, sizeMatrix, tElapsed, "outer loop");

  //daten vom devices kopieren!!!
  h_outputMatrix = (float*)malloc(memSizeMatrix);
  cudaErr(cudaMemcpy(h_outputMatrix, d_outMatrix, memSizeMatrix, cudaMemcpyDeviceToHost));

  //Matrix testen
  char name1[25] = "CPU ";
  strcat(name1,funcName);
  char name2[25] = "GPU ";
  strcat(name2,funcName);
  compareMatrix(ctrMatrix, h_outputMatrix, sizeMatrix, name1, name2);

  //DevicesSpeicher freigeben
  cudaErr(cudaFree(d_inMatrix));
  cudaErr(cudaFree(d_outMatrix));
  //Hostspeicher freigeben
  free(h_outputMatrix);
}
/*
  Kernel invocation and time measure with OMP_GET_WTIME
*/
__host__ void measureKernelOMP(float* inputMatrix, float* ctrMatrix, dim3 gridDim, dim3 blockDim, int matrixWidth, int matrixHeight, int nReps, char *funcName, _kernel_ func){
  int memSizeMatrix = 0;
  int sizeMatrix = 0;
  double tStart = 0;
  double tStop = 0;
  float *d_inMatrix; //input Matrix for parallel gpu copy kernel
  float *d_outMatrix; //output Matrix for parallel gpu copy kernel
  float *h_outputMatrix;

  char processName[50] = "GPU ";
  strcat(processName, funcName);
  strcat(processName, " with omp_get_wtime()");

  preProcess( processName  );
  sizeMatrix = matrixWidth * matrixHeight;
  memSizeMatrix = sizeMatrix * sizeof(float);

  //device mem alloc
  cudaErr(cudaMalloc((void**)&d_inMatrix, memSizeMatrix));//MemSize anders bestimmen
  cudaErr(cudaMalloc((void**)&d_outMatrix, memSizeMatrix));

  //InMatrix zum devices Kopieren!
  cudaErr(cudaMemcpy(d_inMatrix, inputMatrix, memSizeMatrix, cudaMemcpyHostToDevice));

  //Warmup round
  func<<<gridDim,blockDim>>>(d_outMatrix, d_inMatrix, matrixWidth, matrixHeight, 1);

  //Loop inside
  tStart = omp_get_wtime();
  for(int i = 0; i < 1; ++i){
    func<<<gridDim,blockDim>>>(d_outMatrix, d_inMatrix, matrixWidth, matrixHeight, nReps);
    cudaErr(cudaDeviceSynchronize());
  }
  tStop = omp_get_wtime();
  postProcessOMP(nReps, sizeMatrix, (tStop-tStart), "inner loop");

  //Loop outside
  tStart = omp_get_wtime();
  for(int i = 0; i< nReps; ++i){
    func<<<gridDim,blockDim>>>(d_outMatrix, d_inMatrix, matrixWidth, matrixHeight, 1);
    cudaErr(cudaDeviceSynchronize());
  }
  tStop = omp_get_wtime();
  postProcessOMP(nReps, sizeMatrix, (tStop-tStart), "outer loop");

  //Daten vom devices kopieren!!!
  h_outputMatrix = (float*)malloc(memSizeMatrix);
  cudaErr(cudaMemcpy(h_outputMatrix, d_outMatrix, memSizeMatrix, cudaMemcpyDeviceToHost));

  //Matrix testen
  char name1[25] = "CPU ";
  strcat(name1,funcName);
  char name2[25] = "GPU ";
  strcat(name2,funcName);
  compareMatrix(ctrMatrix, h_outputMatrix, sizeMatrix, name1, name2);

  //DevicesSpeicher freigeben
  cudaErr(cudaFree(d_inMatrix));
  cudaErr(cudaFree(d_outMatrix));
  //Hostspeicher freigeben
  free(h_outputMatrix);
}
