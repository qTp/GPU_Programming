#include "../header/cudatool.h"

void cudaErr(cudaError_t err){
    if (err != cudaSuccess){
      printf( "Cuda error in file '%s' in line %i : %s.",
      __FILE__, __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
		}
}
