#ifndef _CUDATOOL_H
#define _CUDATOOL_H

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void cudaErr(cudaError_t err);

#endif /* _CUDATOOL_H */
