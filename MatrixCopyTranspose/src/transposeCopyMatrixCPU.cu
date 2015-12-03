#include "../header/transposeCopyMatrixCPU.h"


void serialCopy(float *outM, float* inM, int width, int height, int nreps){
  int size = width * height;
  for(int measure = 0; measure < nreps; ++measure){
    for(int i = 0; i < size; ++i){
      outM[i] = inM[i];
    }
  }
}
//haha...das ist kein transpose ... ich drehe einfach nur den speicher um =))
void serialTranspose( float* outM, float *inM, int width, int height, int nreps){
  for(int measure = 0; measure < nreps; ++measure){
    for(int j = 0; j < height; ++j){
      for(int i = 0; i < width; i++){
        outM[j * height + i] = inM[j + width * i];
      }
    }
  }
}

void measureAndBuildserialCopy(float*outM, float* inM, int width, int height, int nReps){
  double tStart = 0;
  double tStop = 0;

  preProcess("CPU serialCopy");

  //Loop inside
  tStart = omp_get_wtime();
  for(int i=0; i < 1; ++i){
    serialCopy(outM, inM, width, height, nReps);
  }
  tStop = omp_get_wtime();
  postProcessOMP(nReps, (width*height) * sizeof(float), (tStop-tStart), "inner loop" );

  //Loop outside
  tStart = omp_get_wtime();
  for(int i = 0; i < nReps; i++){
    serialCopy(outM, inM, width, height, 1);
  }
  tStop = omp_get_wtime();
  postProcessOMP(nReps, (width*height) * sizeof(float), (tStop-tStart), "outer loop" );

  //Ergebnis auch testen!
  compareMatrix(inM, outM, (width*height), "inputMatrix", "serialCopy");
  //ende
}

void measureAndBuildserialTranspose(float*outM, float* inM, int width, int height, int nReps){
  int memSize = 0;
  double tStart = 0;
  double tStop = 0;
  float *ctr2Matrix; // compare and proof Matrix

  preProcess("CPU serialTranspose");
  memSize = sizeof(float) * (width*height);
  ctr2Matrix   = (float*)malloc(memSize);

  //Loop inside
  tStart = omp_get_wtime();
  for(int i=0; i<1; ++i){
    serialTranspose(outM, inM, width, height, nReps);
  }
  tStop = omp_get_wtime();
  postProcessOMP( nReps, memSize, (tStop - tStart), "inner loop" );

  //Loop outside
  tStart = omp_get_wtime();
  for (int i = 0; i < nReps; ++i){
    serialTranspose(outM, inM, width, height, 1);
  }
  tStop = omp_get_wtime();
  postProcessOMP( nReps, memSize, (tStop - tStart), "outer loop" );

  //Ergebnis auch testen!
  //nach zweitem Transpose muss Matrix wieder im Original zustand sein!
  serialTranspose(ctr2Matrix, outM, width, height,1);
  compareMatrix(inM, ctr2Matrix, (width*height), "inputMatrix", "2xserialTranspose");

  free(ctr2Matrix);
  //ende
}
