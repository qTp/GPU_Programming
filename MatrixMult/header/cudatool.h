#ifndef _CUDATOOL_H
#define _CUDATOOL_H

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define cudaErr(err) cudaErrT(err, __LINE__,__FILE__)

void cudaErrT(cudaError_t err, int line, char* file );

#endif /* _CUDATOOL_H */
