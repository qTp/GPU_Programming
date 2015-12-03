#ifndef TRANSPOSECOPYMATRIX_CPU_H_
#define TRANSPOSECOPYMATRIX_CPU_H_

#include "../header/cudatool.h"

void serialCopy(float *outM, float* inM, int widht, int height, int nreps);
void serialTranspose( float* outM, float *inM, int width, int height, int nreps);
void measureAndBuildserialCopy(float*outM, float* inM, int width, int height, int nReps);
void measureAndBuildserialTranspose(float* outM, float* inM, int width, int height, int nReps);

#endif /* TRANSPOSECOPYMATRIX_CPU_H_ */
