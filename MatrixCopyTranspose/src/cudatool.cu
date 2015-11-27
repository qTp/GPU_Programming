#include "../header/cudatool.h"

//ErrorHandling for CUDA functions
cudaError_t cudaErrT(cudaError_t err, int line, char* file){
 //, int line, char* file){
#if defined(DEBUG) || defined(_DEBUG)
    if (err != cudaSuccess){
      printf( "\n*** Cuda error in file '%s' in line %i : %s. ***\n\n", file, line, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
		}
#endif
    return err;
}

//Matrix gegen einander testen
void compareMatrix(double *P1, double *P2, int N, char name1[25], char name2[25]){
#if defined(DEBUG)||defined(_DEBUG)
	double epsilon = 1.0e-8; //Fehlertoleranz
	// int match = 1;
	for (int i = 0; i < N; ++i){
		if (abs(P1[i] - P2[i]) > epsilon){
			// match = 0;
			printf("Arrays do not match!\nCompare between %s & %s!\n", name1, name2);
			printf("host:%5.10f gpu:%5.10f at Element %d\n\n", P1[i], P2[i], i);
			break;
		}
	}
	//if (match) printf("Arrays match! Compare between %s & %s.\n\n", name1, name2);
#endif
}

//Matrix gegen einander testen
void compareMatrix(float *P1, float *P2, int N, char name1[25], char name2[25]){
#if defined(DEBUG)||defined(_DEBUG)
	double epsilon = 1.0e-8; //Fehlertoleranz
	// int match = 1;
	for (int i = 0; i < N; ++i){
		if (abs(P1[i] - P2[i]) > epsilon){
			// match = 0;
			printf("Arrays do not match!\nCompare between %s & %s!\n", name1, name2);
			printf("host:%5.10f gpu:%5.10f at Element %d\n\n", P1[i], P2[i], i);
			break;
		}
	}
	// if (match) printf("Arrays match! Compare between %s & %s.\n\n", name1, name2);
#endif
}

//Matrix fuellen
void initMatrix(float *ip, int size){
	//random seed erstellen
	time_t t;
	srand((unsigned)time(&t));
	//Matrix auffuellen
	for (int i = 0; i < size; ++i){
		ip[i] = (float)(rand() & 0xFF) / 100.0f;
	}
}

//Ausgabe welcher Teil gestartet wird
void preProcess(char _name[40]){
  printf("\n*** Starte %s ***\n", _name );
}
//Ausgabe der Ergebnisse
void postProcess(int sumReps, int size, double tElapsed){
    printf("Time elapsed: %.5f ms\n", 1e3*(tElapsed / sumReps ));
    printf("Bandwidth: %.5f GB/s\n", (( (2 * sizeof(float) * size * sumReps) / (1024 * 1024 * 1024)) / tElapsed ) );
}
