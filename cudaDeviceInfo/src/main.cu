/************************************************************/
/*little CommandLinie Tool for reading GPU CUDA informations*/
/************************************************************/

#include "../header/cudatool.h"

inline int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
        { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
        { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
        {   -1, -1 }
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
    return nGpuArchCoresPerSM[index-1].Cores;
}

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
		printf("memoryClockRate: %d | memoryBusWidth: %d\n",dev_prop->memoryClockRate, dev_prop->memoryBusWidth);
		printf("maxThreadsPerMultiProcessor: %d\n", dev_prop->maxThreadsPerMultiProcessor);
		printf("multiProcessorCount: %d\n", dev_prop->multiProcessorCount);
		printf("SM count: %d\n", _ConvertSMVer2Cores(dev_prop->major,dev_prop->minor));
	}

	free(dev_prop);
	cudaDeviceReset();
	return (0);
}
