#include "../header/cudatool.h"
#include "../header/matrixmult_gpu.h"

//Standard version of MatrixMult on the CUDA GPU
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
//Version with shared memory access of MatrixMult on the CUDA GPU
__global__ void cudaMatrixMultWithSMem(float *d_M, float *d_N, double *d_P, int width){
	//TODO TILE_WIDTH als Parameter mit geben!!
	//TODO ablauf pruefen!! Ergebnismatrix ist leer!! wahrscheinlich Grenzen nicht richtig bestimmt!
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

	int blkX = blockIdx.x; int blkY = blockIdx.y;
	int thdX = threadIdx.x; int thdY = threadIdx.y;

	//ROW und COLUMN der Ergebnismatrix bestimmen
	int rowP = blkY * TILE_WIDTH + thdY;
	int colP = blkX * TILE_WIDTH + thdX;

	double pValue = 0;

	if( (rowP < width) && (colP < width)){
		//schleife ueber die d_M und d_N TILES, zum bestimmen des ErgebnisTILES
		//OuterLoop geht die einzelen TILES ab ;-)
		for	( int m = 0; m < width/TILE_WIDTH; m++){

			int globalAccesM = rowP*width+(m*TILE_WIDTH + thdX);
			int globalAccesN = colP+(m*TILE_WIDTH + thdY)*width;

			//Alle Threads in diesem Block!! Versorgen den Speicher mit d_M und d_N Elementen
			Mds[thdY][thdX] = d_M[globalAccesM];
			Nds[thdY][thdX] = d_N[globalAccesN];
			__syncthreads();

			//Nach dem alle Threads im Block die Daten geladen haben wird jetzt die erste P Tile bestimmt
			for( int k = 0; k < TILE_WIDTH; ++k){
				pValue += Mds[thdY][k] * Nds[k][thdX];
			}
			__syncthreads();
			d_P[rowP*width+colP] = pValue;
		}
	}
}
