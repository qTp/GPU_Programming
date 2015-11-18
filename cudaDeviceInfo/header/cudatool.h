#ifndef _CUDATOOL_H_
#define _CUDATOOL_H_

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void cudaErr(cudaError_t err);

#endif
