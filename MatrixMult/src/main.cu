#include "../header/main.h"

//Matrix fuellen
void initMatrix(float *ip, int size){
	//random seed erstellen
	time_t t;
	srand((unsigned)time(&t));
	//Matrix auffuellen
	for (int i = 0; i < size; ++i){
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
	}
}

//Matrix gegen einander testen
void checkMatrix(double *Pserial, double *Pkernel, int N, char string[40]){
	double epsilon = 1.0e-8; //Fehlertoleranz
	int match = 1;
	for (int i = 0; i < N; ++i){
		if (abs(Pserial[i] - Pkernel[i]) > epsilon){
			match = 0;
			printf("Arrays do not match between %s!\n", string);
			printf("host:\n%5.10f\ngpu:\n%5.10f\nat Element %d\n", Pserial[i], Pkernel[i], i);
			break;
		}
	}
	if (match) printf("Arrays match between %s.\n\n", string);
}

int main(int argc, char **argv){

	int h_width;//Breite der Matrix
	int h_arraySize;// groesse der Matrix = width * width
	int memSize = 0;
	int memSizeErg = 0;
	char cBetween[20];
	//variable for time calc
	double tStart = 0;
	double tEnd = 0;

	float *h_M;//Matrix 1 -> host = cpu
	float *h_N;//Matrix 2 -> host = cpu
	float *d_M;//Matrix 1 -> device = gpu <- im Devices-Speicher!
	float *d_N;//Matrix 2 -> device = gpu <- im Devices-Speicher!
	double *h_Ps;//Ergebnis serial
	double *h_Pk;//Ergebnis kernel without shared mem
	double *h_PkSmem;//Ergebnis kernel with shared mem
	double *d_Pk;//Ergebnis kernel <- im Devices-Speicher!
	dim3 dimGrid, dimBlock, dimGridSMEM, dimBlockSMEM;

	if(argc == 4){
		h_width = atoi(argv[1]);
		dimBlock.x = atoi(argv[2]);
		dimBlock.y = atoi(argv[3]);
		dimBlock.z = 1;
		//Aus Block Grid bestimmen.
		dimGrid.x = ceil((float) h_width / dimBlock.x);
		dimGrid.y = ceil((float) h_width / dimBlock.y);
		dimGrid.z = 1;

		dimBlockSMEM.x = TILE_WIDTH;
		dimBlockSMEM.y = TILE_WIDTH;
		dimBlockSMEM.z = 1;

		dimGridSMEM.x = ceil((float)h_width / TILE_WIDTH);
		dimGridSMEM.y = ceil((float)h_width / TILE_WIDTH);
		dimGridSMEM.z = 1;

	}else{
		printf("Falsche Parameter Anzahl!\nwidth blockX blockY!\n\n");
		exit(-1);
	}

	//Bestimmen der KERNEL Parameter ( GRID Dim & BLOCK Dim )
	printf("Normal: Grid: %d|%d|%d and Block: %d|%d|%d for width: %d.\n", dimGrid.x,dimGrid.y,dimGrid.z, dimBlock.x, dimBlock.y,dimBlock.z, h_width);
	printf("SMem: Grid: %d|%d|%d and Block: %d|%d|%d for width: %d.\n", dimGridSMEM.x,dimGridSMEM.y,dimGridSMEM.z, dimBlockSMEM.x, dimBlockSMEM.y,dimBlockSMEM.z, h_width);

	//mit der width die Array groesse berechen
	h_arraySize = h_width * h_width;

	//Speichergroesse bestimmen
	memSize = sizeof(float)*h_arraySize;
	memSizeErg = sizeof(double)*h_arraySize;

	//Host Arrays allokieren
	h_M = (float*)malloc(memSize);
	h_N = (float*)malloc(memSize);

	//Host-Ergebnis Array initialisieren
	h_Ps = (double*)malloc(memSizeErg);
	h_Pk = (double*)malloc(memSizeErg);
	h_PkSmem = (double*)malloc(memSizeErg);

	//devices array allokieren
	cudaErr(cudaMalloc((void**)&d_M, memSize));
	cudaErr(cudaMalloc((void**)&d_N, memSize));

	//Device-Ergebnis array initialisieren
	cudaErr(cudaMalloc((void**)&d_Pk, memSizeErg));

	//Matrix mit zufaelligen Werten fuellen
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
/*
	//Matrix auf der GPU berechnen; am besten diverse GRID | BLOCK kompiationen Testen
	printf("Start GPU MatrixMult aufwaermen\n");
	tStart = omp_get_wtime();
	cudaMatrixMult<<<dimGrid, dimBlock>>>(d_M, d_N, d_Pk, h_width);
	cudaErr(cudaDeviceSynchronize());
	tEnd = omp_get_wtime();
	printf("Finish GPU MatrixMult in %f ms\n\n", 1.e3*(tEnd - tStart));

	printf("Start GPU MatrixMult jetzt aber richtig.\n");
	tStart = omp_get_wtime();
	cudaMatrixMult<<<dimGrid, dimBlock>>>(d_M, d_N, d_Pk, h_width);
	cudaErr(cudaDeviceSynchronize());
	tEnd = omp_get_wtime();
	printf("Finish GPU MatrixMult in %f ms\n\n", 1.e3*(tEnd - tStart));
	//Ergebnis kopieren
	cudaErr(cudaMemcpy(h_Pk, d_Pk, memSizeErg, cudaMemcpyDeviceToHost));
	//Matrix testen
	strcpy(cBetween, "CPU - GPU w\\o smem");
	checkMatrix(h_Ps, h_Pk, h_arraySize,cBetween);
*/
	printf("Start GPU MatrixMult SharedMem aufwaermen.\n");
	tStart = omp_get_wtime();
	cudaMatrixMultWithSMem<<<dimGridSMEM, dimBlockSMEM>>>(d_M, d_N, d_Pk, h_width);
	cudaErr(cudaDeviceSynchronize());
	tEnd = omp_get_wtime();
	printf("Finish GPU MatrixMult SharedMem in %f ms\n\n", 1.e3*(tEnd - tStart));

	printf("Start GPU MatrixMult SharedMem jetzt aber richtig..\n");
	tStart = omp_get_wtime();
	cudaMatrixMultWithSMem<<<dimGridSMEM, dimBlockSMEM>>>(d_M, d_N, d_Pk, h_width);
	cudaErr(cudaDeviceSynchronize());
	tEnd = omp_get_wtime();
	printf("Finish GPU MatrixMult SharedMem in %f ms\n\n", 1.e3*(tEnd - tStart));
	//Ergebnis kopieren
	cudaErr(cudaMemcpy(h_PkSmem, d_Pk, memSizeErg, cudaMemcpyDeviceToHost));
	//Matrix testen
	strcpy(cBetween, "CPU - GPU with smem");
	checkMatrix(h_Ps, h_PkSmem, h_arraySize, cBetween);


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
