#include "../header/cudatool.h"
#include "../header/matrixmult_cpu.h"

//serial MatrixMult
void serialMatrixMult(float *M, float *N, double *P, int width){
	for (int i = 0; i < width; ++i){
		for (int j = 0; j < width; ++j){
			double sum = 0;
			for (int k = 0; k < width; ++k){
				sum +=  M[i*width + k] * N[k*width + j];
			}
			P[i*width + j] = sum;
		}
	}
}
