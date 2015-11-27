#include "../header/main.h"

int main(int argc, char**argv){
  int matrixWidth = 0;
  int matrixHeight = 0;
  int innerReps = 250; //inner 	repetition: if outerReps >= 1 then innerReps = 1
  int outerReps = 1; // outer 	repetition: if innerReps >= 1 then outerReps = 1

  float *inputMatrix; //inputMatrix
  float *ctrTMatrix; // compare and proof MatrixTranspose
  float *ctrCMatrix; // compare and proof MatrixCopy


  if(argc == 4){ // 0:_Name; 1:width; 2:height; 3:blockDim;
    matrixWidth = atoi(argv[1]);
    matrixHeight = atoi(argv[2]);

    dim3 blockDim = {1,1,1};
    dim3 gridDim = {1,1,1};

    printf("\nMatrix dimension: width=%d, height=%d\n", matrixWidth, matrixHeight);
    printf("Wiederholungen: outerFunc=%d, innerFunc=%d\n", outerReps, innerReps);
    printf("GPU Device: %s\n", "devicePropsAuslesen!!!");
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

  //Aufraeumen nicht vergessen!!!
  free(ctrTMatrix);
  free(ctrCMatrix);
  free(inputMatrix);
  //GPU Device zuruecksetzten
  cudaDeviceReset();
  //OS mitteilen das wir durch sind.
  return EXIT_SUCCESS;
}
