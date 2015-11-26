#include "../header/cudatool.h"

cudaError_t cudaErrT(cudaError_t err, int line, char* file){
 //, int line, char* file){
#if defined(DEBUG) || defined(_DEBUG)
    if (err != cudaSuccess){
      printf( "\n*** Cuda error in file '%s' in line %i : %s. ***\n\n", file, line, cudaGetErrorString(err));
      assert(err != cudaSuccess);
		}
#endif
    return err;
}
