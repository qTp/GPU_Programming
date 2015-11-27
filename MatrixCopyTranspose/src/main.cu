#include "../header/main.h"

int main(int argc, char**argv){
  int matrixWidth = 0;
  int matrixHeight = 0;

  int innerReps = 250; //inner 	repetition: if outerReps >= 1 then innerReps = 1
  int outerReps = 1; // outer 	repetition: if innerReps >= 1 then outerReps = 1

  dim3 blockDim(0);
  dim3 gridDim(0);
  float *inputMatrix; //inputMatrix
  float *ctrTMatrix; // compare and proof MatrixTranspose
  float *ctrCMatrix; // compare and proof MatrixCopy

  //DevicesProperties auslesen
  int deviceCount = 0;
  cudaDeviceProp *dev_prop;
  cudaErr(cudaGetDeviceCount(&deviceCount));
  dev_prop = (cudaDeviceProp*) malloc (deviceCount*sizeof(cudaDeviceProp));
  for(int i=0; i<deviceCount;i++){
    cudaErr(cudaGetDeviceProperties(&dev_prop[i],i));
  }
  //ende

  if(argc == 4){ // 0:_Name; 1:width; 2:height; 3:blockDim;
    matrixWidth = atoi(argv[1]);
    matrixHeight = atoi(argv[2]);

    blockDim.x = TILE_DIM;
    blockDim.y = TILE_DIM;
    blockDim.z = 1;

    gridDim.x = matrixWidth/TILE_DIM;
    gridDim.y = matrixWidth/TILE_DIM;
    gridDim.z = 1;

    printf("\nMatrix dimension: width=%d, height=%d\n", matrixWidth, matrixHeight);
    printf("Wiederholungen: outerFunc=%d, innerFunc=%d\n", outerReps, innerReps);
    printf("GPU Device: %s\n", dev_prop->name );
    printf("Grid dimension: x=%d, y=%d, z=%d\n",gridDim.x, gridDim.y, gridDim.z);
    printf("Block dimension: x=%d, y=%d, z=%d\n",blockDim.x, blockDim.y, blockDim.z);

  }else{
    printf("Wrong paramter!\n\n");
    exit(EXIT_FAILURE);
  }

  //speicherbedarf bestimmen
  int sizeMatrix = matrixWidth * matrixHeight ;
  int memSizeMatrix = ( sizeof(float) * ( matrixWidth * matrixHeight ));
  //Speicher holen
  inputMatrix = (float*)malloc(memSizeMatrix);
  ctrTMatrix = (float*)malloc(memSizeMatrix);
  ctrCMatrix = (float*)malloc(memSizeMatrix);

  //create inputMatrix
  initMatrix(inputMatrix, sizeMatrix);
  //create control Matrix for TRANSPOSE test!
  measureAndBuildserialTranspose(ctrTMatrix, inputMatrix, sizeMatrix, outerReps, innerReps);
  //create control Matrix for COPY test!
  measureAndBuildserialCopy(ctrCMatrix, inputMatrix, sizeMatrix, outerReps, innerReps);

//
//parallel copy kernel invocation
int sumReps = 0;
double tStart = 0;
double tStop = 0;
float *d_inCopyMatrix; //input Matrix for parallel gpu copy kernel
float *d_outCopyMatrix; //output Matrix for parallel gpu copy kernel
float *h_outputCopyMatrix;
preProcess("gpu parallel copyMatrix");
//device mem alloc
cudaErr(cudaMalloc((void**)&d_inCopyMatrix, memSizeMatrix));//MemSize anders bestimmen
cudaErr(cudaMalloc((void**)&d_outCopyMatrix, memSizeMatrix));
//Daten zum devices Kopieren!
//InMatrix zum devices kopieren
cudaErr(cudaMemcpy(d_inCopyMatrix, inputMatrix, memSizeMatrix, cudaMemcpyHostToDevice));
tStart = omp_get_wtime();
for(int i = 0; i<outerReps; ++i){
    copyMatrix<<<gridDim,blockDim>>>(d_outCopyMatrix,d_inCopyMatrix,matrixWidth,matrixHeight,innerReps);
    cudaErr(cudaDeviceSynchronize());
  }
tStop = omp_get_wtime();
postProcess(sumReps, sizeMatrix, (tStop-tStart));
//daten vom devices kopieren!!!
h_outputCopyMatrix = (float*)malloc( memSizeMatrix);
cudaErr(cudaMemcpy(h_outputCopyMatrix, d_outCopyMatrix, memSizeMatrix, cudaMemcpyDeviceToHost));
//DevicesSpeicher freigeben
cudaErr(cudaFree(d_inCopyMatrix));
cudaErr(cudaFree(d_outCopyMatrix));
compareMatrix(ctrCMatrix, h_outputCopyMatrix, sizeMatrix, "CPU CopyMatrix","GPU CopyMatrix" );
//
//
  //Aufraeumen nicht vergessen!!!
  free(ctrTMatrix);
  free(ctrCMatrix);
  free(inputMatrix);
  free(dev_prop);
  //GPU Device zuruecksetzten
  cudaDeviceReset();
  //OS mitteilen das wir durch sind.
  return EXIT_SUCCESS;
}
