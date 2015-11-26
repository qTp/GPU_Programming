#include "../header/main.h"

int main(int argc, char**argv){
  float *dPtr;
  cudaErr(cudaMalloc((void**)&dPtr, sizeof(float) * 100 ));

  cudaErr(cudaFree(dPtr));
  cudaErr(cudaFree(dPtr));

}
