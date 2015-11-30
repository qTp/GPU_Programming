#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void checkIndex(void){
	printf("threadIdx: (%d, %d, %d blockIdx: (%d, %d, %d) gridIdx: (%d, %d, %d) blockDim: (%d, %d, %d) gridDim: (%d, %d, %d)\n",
	threadIdx.x, threadIdx.y, threadIdx.z,
	blockIdx.x, blockIdx.y, blockIdx.z,
	gridIdx.x, gridIdx.y, gridIdx.z,
	blockDim.x, blockDim.y, blockDim.z,
	gridDim.x, gridDim.y, gridDim.z);
}

__host__ int main(int args, char **argv){
	//define toatl data element
	int nElem = 6;

	//define grid and block structure
	dim3 block (3,3);
	dim3 grid(3,3);

	//check grid and block dimension from host side
	printf("grid.x %d, grid.y %d, grid.z %d\n", grid.x, grid.y, grid.z);
	printf("block.x %d, block.y %d, block.z %d\n", block.x, block.y, block.z);

	//check dimension from device side
	checkIndex<<<grid, block>>>();

	//rest device before you leave
	cudaDeviceReset();

	return 0;
}
