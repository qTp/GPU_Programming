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
    for(int i = 0; i < size; ++i){
      outM[size-i-1] = inM[i];
    }
  }
}

void measureAndBuildserialCopy(float*outM, float* inM, int size, int outerReps, int innerReps){
  int sumReps = 0;
  double tStart = 0;
  double tStop = 0;

  preProcess("serialCopy");
  sumReps = outerReps * innerReps;
  tStart = omp_get_wtime();
  for(int i = 0; i < outerReps; ++i){
    serialCopy(outM, inM, size, innerReps);
  }
  tStop = omp_get_wtime();
  postProcess(sumReps,size,(tStop-tStart) );
  compareMatrix(inM, outM, size, "inputMatrix", "serialCopy");
}

void measureAndBuildserialTranspose(float*outM, float* inM, int size, int outerReps, int innerReps){
  int sumReps = 0;
  double tStart = 0;
  double tStop = 0;
  float *ctr2Matrix; // compare and proof Matrix

  preProcess("serialTranspose");
  ctr2Matrix   = (float*)malloc(sizeof(float) * size);
  sumReps = outerReps * innerReps;
  tStart = omp_get_wtime();
  for (int i = 0; i < outerReps; ++i){
    serialTranspose(outM, inM, size, innerReps);
  }
  tStop = omp_get_wtime();
  postProcess( sumReps, size, (tStop - tStart) );
  //nach zweitem Transpose muss Matrix wieder im Original zustand sein!
  serialTranspose(ctr2Matrix, outM, size,1);
  compareMatrix(inM, ctr2Matrix, size, "inputMatrix", "2xserialTranspose");
  free(ctr2Matrix);
}
