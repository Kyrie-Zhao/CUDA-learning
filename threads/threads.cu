#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdio>

__global__ void threads(){
    printf("threadID.x: %d, threadID.y: %d, threadID.y: %d\n",threadIdx.x,threadIdx.y,threadIdx.z);
    printf("blockID.x: %d, blockID.y: %d, blockID.y: %d\n",blockIdx.x,blockIdx.y,blockIdx.z);
    printf("blockDim.x: %d, blockDim.y: %d, gridDim.x: %d, gridDim.y: %d\n\n",blockDim.x,blockDim.y,gridDim.x,gridDim.y);
}

int main(void){
    int nx, ny;
    nx = 2;
    ny = 2;
    dim3 block(1,1);
    dim3 grid(nx/block.x,ny/block.y);

    threads<<<block,grid>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();

}