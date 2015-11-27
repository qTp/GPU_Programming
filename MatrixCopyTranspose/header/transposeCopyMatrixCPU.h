#ifndef TRANSPOSECOPYMATRIX_CPU_H_
#define TRANSPOSECOPYMATRIX_CPU_H_

#include "../header/cudatool.h"

void serialCopy(float *outM, float* inM, int size, int nreps);
void serialTranspose( float* outM, float *inM, int size, int nreps);
void measureAndBuildserialCopy(float*outM, float* inM, int size, int outerReps, int innerReps);
void measureAndBuildserialTranspose(float* outM, float* inM, int size, int outerReps, int innerReps);

#endif /* TRANSPOSECOPYMATRIX_CPU_H_ */
