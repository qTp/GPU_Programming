
#include <stdbool.h>

#include <math.h>
#include <time.h>
#include <omp.h>

#include "../header/cudatool.h"

#define BLOCK_WIDTH 16


//parallel MatrixMult
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

//Matrix fuellen
void initMatrix(float *ip, int size){
	//random seed erstellen
	time_t t;
	srand((unsigned)time(&t));
	//Matrix auffuellen
	for (int i = 0; i < size; i++){
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
	}
}

//Matrix gegen einander testen
bool checkMatrix(double *Pserial, double *Pkernel, int N){
	double epsilon = 1.0e-8; //Fehlertoleranz
	bool match = 1;
	for (int i = 0; i < N; i++){
		if (abs(Pserial[i] - Pkernel[i]) > epsilon){
			match = 0;
			printf("Arrays do not match!\n");
			printf("host:\n%5.10f\ngpu:\n%5.10f\nat Element %d\n", Pserial[i], Pkernel[i], i);
			break;
		}
	}
	if (match) printf("Arrays match.\n\n");
	return match;
}

int main(int argc, char **argv){
	int h_width;//Breite der Matrix
	int h_arraySize;// groesse der Matrix = width * width
	int memSize = 0;
	int memSizeErg = 0;

	//variable for time calc
	double tStart = 0;
	double tEnd = 0;

	float *h_M;//Matrix 1 -> host = cpu
	float *h_N;//Matrix 2 -> host = cpu
	float *d_M;//Matrix 1 -> device = gpu <- im Devices-Speicher!
	float *d_N;//Matrix 2 -> device = gpu <- im Devices-Speicher!
	double *h_Ps;//Ergebnis serial
	double *h_Pk;//Ergebnis kernel
	double *d_Pk;//Ergebnis kernel <- im Devices-Speicher!
	dim3 dimGrid, dimBlock;

	if(argc == 4){
		h_width = atoi(argv[1]);
		dimBlock.x = atoi(argv[2]);
		dimBlock.y = atoi(argv[3]);
		dimBlock.z = 1;
		//Aus Block Grid bestimmen.
		dimGrid.x = ceil((float) h_width / dimBlock.x);
		dimGrid.y = ceil((float) h_width / dimBlock.y);
		dimGrid.z = 1;

	}else{
		printf("Falsche Parameter Anzahl!\n width blockX blockY !");
		exit(-1);
	}

	//Bestimmen der KERNEL Parameter ( GRID Dim & BLOCK Dim )
//	numBlocks = ceil((float) h_width / BLOCK_WIDTH );
	printf("Blocks= %d|%d for w: %d , Grid: %d|%d\n", dimBlock.x, dimBlock.y, h_width, dimGrid.x,dimGrid.y);
	//dim3 dimGrid(numBlocks, numBlocks);
	//dim3 dimBlock(BLOCK_WIDTH,BLOCK_WIDTH);

	//mit der width die Array groesse berechen
	h_arraySize = h_width * h_width;

	//Speichergroesse bestimmen
	memSize = (sizeof(float)*h_arraySize);
	memSizeErg = sizeof(double)*h_arraySize;
	//Host Arrays allokieren
	h_M = (float*)malloc(memSize);
	h_N = (float*)malloc(memSize);

	//Host-Ergebnis Array initialisieren
	h_Ps = (double*)malloc(memSizeErg);
	h_Pk = (double*)malloc(memSizeErg);

	//devices array allokieren
	cudaErr(cudaMalloc((void**)&d_M, memSize));
	cudaErr(cudaMalloc((void**)&d_N, memSize));

	//Device-Ergebnis array initialisieren
	cudaErr(cudaMalloc((void**)&d_Pk, memSizeErg));

	//Matrix mit zuf�lligen Werten fuellen
	initMatrix(h_M, h_arraySize);
	initMatrix(h_N, h_arraySize);

	//host array in Device-Array kopieren
	cudaErr(cudaMemcpy(d_M, h_M, memSize, cudaMemcpyHostToDevice));
	cudaErr(cudaMemcpy(d_N, h_N, memSize, cudaMemcpyHostToDevice));

	//Matrix zu Testzwecken serielle berchenen lassen & Zeitmessung
	//Zeitmessung implementieren!!!
	printf("Start CPU MatrixMult\n");
	tStart = omp_get_wtime();
	serialMatrixMult(h_M, h_N, h_Ps, h_width);
	tEnd = omp_get_wtime();
	printf("Finish CPU MatrixMult in %f ms\n\n",1.e3*(tEnd - tStart));

	//Matrix auf der GPU berechnen; am besten diverse GRID | BLOCK kompiationen Testen
	printf("Start GPU MatrixMult aufwaermen\n");
	tStart = omp_get_wtime();
	cudaMatrixMult<<<dimGrid, dimBlock>>>(d_M, d_N, d_Pk, h_width);
	cudaErr(cudaDeviceSynchronize());
	tEnd = omp_get_wtime();
	printf("Finish GPU MatrixMult in %f ms\n\n", 1.e3*(tEnd - tStart));

	printf("Start GPU MatrixMult jetzt aber richtig XD\n");
	tStart = omp_get_wtime();
	cudaMatrixMult<<<dimGrid, dimBlock>>>(d_M, d_N, d_Pk, h_width);
	cudaErr(cudaDeviceSynchronize());
	tEnd = omp_get_wtime();
	printf("Finish GPU MatrixMultin %f ms\n\n", 1.e3*(tEnd - tStart));

	//Ergebnis kopieren
	cudaErr(cudaMemcpy(h_Pk, d_Pk, memSizeErg, cudaMemcpyDeviceToHost));

	//Matrix testen
	checkMatrix(h_Ps, h_Pk, h_arraySize);

	//Alles befreien
	free(h_M);
	free(h_N);
	free(h_Ps);
	cudaErr(cudaFree(d_M));
	cudaErr(cudaFree(d_N));
	cudaErr(cudaFree(d_Pk));

	//nicht vergessen ;-)
	cudaDeviceReset();
	//Programm mit Erfolg beenden
	return 0;
}
