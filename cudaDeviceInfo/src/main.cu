/************************************************************/
/*little CommandLinie Tool for reading GPU CUDA informations*/
/************************************************************/

#include "../header/cudatool.h"

int main(int argc, char **argv){
	int dev_count = 0;
	cudaDeviceProp *dev_prop;
	cudaErr(cudaGetDeviceCount(&dev_count));
	printf("We got %d device(s)\n", dev_count);
	dev_prop = (cudaDeviceProp*) malloc (dev_count*sizeof(cudaDeviceProp));

	for(int i=0; i<dev_count;i++){
		cudaErr(cudaGetDeviceProperties(&dev_prop[i],i));
		printf("Following properties are read for the device: %d\n", i+1);
		printf("Name: %s\nGlobalMem: %Iu\n", dev_prop->name, -1e-6*dev_prop->totalGlobalMem);
		printf("sharedMemPerBLock: %Iu\tregsPerBlock: %d\n", dev_prop->sharedMemPerBlock, dev_prop->regsPerBlock);
		printf("warpSize: %d\nmemPitch: %Iu\n", dev_prop->warpSize, dev_prop->memPitch);
		printf("clockRate: %d\ntotalConstMem: %Iu\n", dev_prop->clockRate, dev_prop->totalConstMem);
		printf("maxThreadsPerBlock: %d\n", dev_prop->maxThreadsPerBlock);
		printf("maxThreadDim[3]: %d %d %d\n", dev_prop->maxThreadsDim[0],dev_prop->maxThreadsDim[1],dev_prop->maxThreadsDim[2]);
		printf("maxGridSize[3]: %d %d %d\n",	dev_prop->maxGridSize[0],dev_prop->maxGridSize[1],dev_prop->maxGridSize[2] );
		printf("totalConstMem: %d\n",	dev_prop->totalConstMem);
		printf("major | minor: %d | %d\n",	dev_prop->major,dev_prop->minor );

	}

	free(dev_prop);
	cudaDeviceReset();
	return (0);
}
