#include "../header/main.h"

int main(int argc, char**argv){
  int matrixWidth = 0;
  int matrixHeight = 0;
  int nReps = 0;
  dim3 blockDim(0);
  dim3 gridDim(0);
  float *inputMatrix; //inputMatrix
  float *ctrTMatrix; // compare and proof MatrixTranspose
  float *ctrCMatrix; // compare and proof MatrixCopy
  float *goldMatrix; // je nach Kernel Func, entweder ctrTMatrix oder ctrCmatrix
  //Kernel function pointer
  _kernel_ kernelFunc;
  //name of the kernel function
  char *funcName;

  //DevicesProperties auslesen
  int deviceCount = 0;
  cudaDeviceProp *dev_prop;
  cudaErr(cudaGetDeviceCount(&deviceCount));
  dev_prop = (cudaDeviceProp*) malloc (deviceCount*sizeof(cudaDeviceProp));
  cudaErr(cudaGetDeviceProperties(dev_prop,0));

  //ende

  if(argc == 4){ // 0:_Name; 1:width; 2:height; 3:NumberOfRepetition;
    matrixWidth = atoi(argv[1]);
    matrixHeight = atoi(argv[2]);
    nReps = atoi(argv[3]);

    blockDim.x = TILE_DIM;
    blockDim.y = BLOCK_ROWS;
    blockDim.z = 1;

    gridDim.x = matrixWidth/(TILE_DIM);
    gridDim.y = matrixWidth/(TILE_DIM);
    gridDim.z = 1;

    printf("\nGPU Device: %s\n", dev_prop->name );
    printf("WarpSize: %d, ThreadsPerMP: %d\n", dev_prop->warpSize, dev_prop->maxThreadsPerMultiProcessor);
    printf("\nMatrix dimension: width=%d, height=%d\n", matrixWidth, matrixHeight);
    printf("Wiederholungen fuer Zeitmessung: %d\n", nReps);
    printf("Grid dimension: x=%d, y=%d, z=%d\n",gridDim.x, gridDim.y, gridDim.z);
    printf("Block dimension: x=%d, y=%d, z=%d\n",blockDim.x, blockDim.y, blockDim.z);

  }else{
    printf("Wrong paramter!\n\n");
    exit(EXIT_FAILURE);
  }
  
  //speicherbedarf bestimmen
  int sizeMatrix = matrixWidth * matrixHeight;
  int memSizeMatrix = ( sizeof(float) * sizeMatrix );

  //Speicher holen
  inputMatrix = (float*)malloc(memSizeMatrix);
  ctrTMatrix = (float*)malloc(memSizeMatrix);
  ctrCMatrix = (float*)malloc(memSizeMatrix);
  //goldMatrix = (float*)malloc(memSizeMatrix);

  //create inputMatrix
  initMatrix(inputMatrix, sizeMatrix);

  //create control Matrix for COPY test!
  measureAndBuildserialCopy(ctrCMatrix, inputMatrix, sizeMatrix, nReps);

  //create control Matrix for TRANSPOSE test!
  measureAndBuildserialTranspose(ctrTMatrix, inputMatrix, sizeMatrix, nReps);

  //schleife ueber alle Kernel Funktionen!!!
  // i = anzahl an KernelFunktionen ;-)
  for(int i = 0; i<1; ++i){
    switch(i){
      case 0:
        //create and measure CopyKernel
        kernelFunc = &copyMatrix;
        funcName = "copyMatrix\0";
        goldMatrix = ctrCMatrix;
        break;
      case 1:
        break;
    }
    //ein call mit der jeweiligen KernelFunc und der ControlMatrix!!!
    measureKernel(inputMatrix, goldMatrix, gridDim, blockDim, matrixWidth, matrixHeight, nReps,funcName, kernelFunc);
  }
  //Aufraeumen nicht vergessen!!!
  free(inputMatrix);
  free(ctrTMatrix);
  free(ctrCMatrix);
  //free(goldMatrix);
  free(dev_prop);
  //GPU Device zuruecksetzten
  cudaDeviceReset();
  //OS mitteilen das wir durch sind.
  return EXIT_SUCCESS;
}
