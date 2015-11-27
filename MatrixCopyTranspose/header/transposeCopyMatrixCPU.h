#ifndef TRANSPOSECOPYMATRIXCPU_H_
#define TRANSPOSECOPYMATRIXCPU_H_

#include "../header/cudatool.h"

void serialCopy(float *outM, float* inM, int size, int nreps);
void serialTranspose( float* outM, float *inM, int size, int nreps);
void measureAndBuildserialCopy(float*outM, float* inM, int size, int outerReps, int innerReps);
void measureAndBuildserialTranspose(float* outM, float* inM, int size, int outerReps, int innerReps);

#endif /* TRANSPOSECOPYMATRIXCPU_H_ */
