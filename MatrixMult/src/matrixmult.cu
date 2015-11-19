#include "../header/cudatool.h"
#include "../header/matrixmult.h"

__global__ void cudaMatrixMult(float *d_M, float *d_N, double *d_P, int width){
	//Berechne die Reihe/Zeile (ROW)
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	//Berechne die Spalte (COLUMN)
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	//Nur berechnen wenn die Zeile/Spalte noch in der Matrix ist! Wichtig!
	//Da hier auch Threads erstellt werden die nicht in der Matrix liegen.
	if ((row < width) && (col < width)){
		double Pvalue = 0;
		//jeder Thread berechnet genau ein Element der Ergebnismatrix
		for (int k = 0; k < width; k++){
			Pvalue += d_M[row*width + k] * d_N[k*width + col];
		}
		d_P[row*width + col] = Pvalue;
	}
}


//serial MatrixMult
void serialMatrixMult(float *M, float *N, double *P, int width){
	for (int i = 0; i < width; i++){
		for (int j = 0; j < width; j++){
			double sum = 0;
			for (int k = 0; k < width; k++){
				sum +=  M[i*width + k] * N[k*width + j];
			}
			P[i*width + j] = sum;
		}
	}
}
