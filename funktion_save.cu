//---------part for header---------------------//
//1D gird 1d block
__device__ int getGlobalIdx_1d_1d();
//1D grid 2D blocks
__device__ int getGlobalIdx_1d_2d();
//1D grid 3D blocks
__device__ int getGlobalIdx_1D_3D();
// 2D grid of 1D blocks
__device__ int getGlobalIdx_2D_1D();
// 2D grid of 2D blocks
__device__ int getGlobalIdx_2D_2D();
// 2D grid of 3D blocks
__device__ int getGlobalIdx_2D_3D();
//3D grid of 1D blocks
__device__ int getGlobalIdx_3D_1D();
//3D grid of 2D blocks
__device__ int getGlobalIdx_3D_2D();
//3D grid of 3D blocks
__device__ int getGlobalIdx_3D_3D();
//----------------------------------------------//
//***********Unique Thread Index****************//

//1D gird 1D BLOCK_ROWS
__device__ int getGlobalIdx_1d_1d(){
  // Im Speicher!!! (BlockIdx * BlockDim)
  // + threadIdx.x = Welches Element vom Speicher bin ich!!!
  return blockIdx.x * blockDim.x + threadIdx.x;
}
//1D grid 2D blocks
__device__ int getGlobalIdx_1d_2d(){
  return blockIdx.x * blockDim.x * blockDim.y
          + threadIdx.y * blockDim.x
          + threadIdx.x;
}
//1D grid 3D blocks
__device__ int getGlobalIdx_1D_3D(){
return blockIdx.x * blockDim.x * blockDim.y * blockDim.z
        + threadIdx.z * blockDim.y * blockDim.x
        + threadIdx.y * blockDim.x
        + threadIdx.x;
}
// 2D grid of 1D blocks
__device__ int getGlobalIdx_2D_1D(){
  int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int threadId = blockId * blockDim.x + threadIdx.x;
  return threadId;
}
// 2D grid of 2D blocks
__device__ int getGlobalIdx_2D_2D(){
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int threadId = blockId * (blockDim.x * blockDim.y)
              + (threadIdx.y * blockDim.x) + threadIdx.x;
  return threadId;
}
// 2D grid of 3D blocks
__device__ int getGlobalIdx_2D_3D(){
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
               + (threadIdx.z * (blockDim.x * blockDim.y))
               + (threadIdx.y * blockDim.x) + threadIdx.x;
  return threadId;
}
//3D grid of 1D blocks
__device__ int getGlobalIdx_3D_1D(){
  int blockId = blockIdx.x + blockIdx.y * gridDim.x
              + gridDim.x * gridDim.y * blockIdx.z;
  int threadId = blockId * blockDim.x + threadIdx.x;
  return threadId;
}
//3D grid of 2D blocks
__device__ int getGlobalIdx_3D_2D(){
  int blockId = blockIdx.x + blockIdx.y * gridDim.x
                + gridDim.x * gridDim.y * blockIdx.z;
  int threadId = blockId * (blockDim.x * blockDim.y)
                + (threadIdx.y * blockDim.x) + threadIdx.x;
  return threadId;
}
//3D grid of 3D blocks
__device__ int getGlobalIdx_3D_3D(){
  int blockId = blockIdx.x
                + blockIdx.y * gridDim.x
                + gridDim.x * gridDim.y * blockIdx.z;
  int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
               + (threadIdx.z * (blockDim.x * blockDim.y))
               + (threadIdx.y * blockDim.x) + threadIdx.x;
  return threadId;
}
