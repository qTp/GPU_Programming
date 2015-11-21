#include "../header/cudatool.h"

void cudaErrT(cudaError_t err, int line, char* file){
    if (err != cudaSuccess){
      printf( "\n*** Cuda error in file '%s' in line %i : %s. ***\n\n",
      file, line, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
		}
}
