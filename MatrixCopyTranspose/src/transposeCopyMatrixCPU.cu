#include "../header/transposeCopyMatrixCPU.h"


void serialCopy(float *outM, float* inM, int size, int nreps){
  for(int measure = 0; measure < nreps; ++measure){
    for(int i = 0; i < size; ++i){
      outM[i] = inM[i];
    }
  }
}

void serialTranspose( float* outM, float *inM, int size, int nreps){
  for(int measure = 0; measure < nreps; ++measure){
    for(int i = 0; i < size; i++){
      outM[size-i-1] = inM[i];
    }
  }
}

void measureAndBuildserialCopy(float*outM, float* inM, int size, int nReps){
  double tStart = 0;
  double tStop = 0;

  preProcess("serialCopy");

  //Loop inside
  tStart = omp_get_wtime();
  serialCopy(outM, inM, size, nReps);
  tStop = omp_get_wtime();
  postProcess(nReps, size * sizeof(float), (tStop-tStart), "inner loop" );

  //Loop outside
  tStart = omp_get_wtime();
  for(int i = 0; i < nReps; i++){
    serialCopy(outM, inM, size, 1);
  }
  tStop = omp_get_wtime();
  postProcess(nReps, size * sizeof(float), (tStop-tStart), "outer loop" );

  //Ergebnis auch testen!
  compareMatrix(inM, outM, size, "inputMatrix", "serialCopy");
  //ende
}

void measureAndBuildserialTranspose(float*outM, float* inM, int size, int nReps){
  int memSize = 0;
  double tStart = 0;
  double tStop = 0;
  float *ctr2Matrix; // compare and proof Matrix

  preProcess("serialTranspose");
  memSize = sizeof(float) * size;
  ctr2Matrix   = (float*)malloc(memSize);

  //Loop inside
  tStart = omp_get_wtime();
  serialTranspose(outM, inM, size, nReps);
  tStop = omp_get_wtime();
  postProcess( nReps, memSize, (tStop - tStart), "inner loop" );

  //Loop outside
  tStart = omp_get_wtime();
  for (int i = 0; i < nReps; ++i){
    serialTranspose(outM, inM, size, 1);
  }
  tStop = omp_get_wtime();
  postProcess( nReps, memSize, (tStop - tStart), "outer loop" );

  //Ergebnis auch testen!
  //nach zweitem Transpose muss Matrix wieder im Original zustand sein!
  serialTranspose(ctr2Matrix, outM, size,1);
  compareMatrix(inM, ctr2Matrix, size, "inputMatrix", "2xserialTranspose");

  free(ctr2Matrix);
  //ende
}
