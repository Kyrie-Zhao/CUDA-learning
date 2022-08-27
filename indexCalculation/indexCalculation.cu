#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdio>

__global__ void indexCalculation_thread(int* input){
    printf("kernel running, fetching via threads\n");
    int tid = threadIdx.x;
    printf("threadIdx : %d, value: %d \n", tid, input[tid]);
}

__global__ void indexCalculation_blockgridThread(int* input){
    printf("kernel running, fetching via gridblock\n");
    int tid = threadIdx.x;
    int offset = blockDim.x*blockIdx.x;
    tid = tid+offset;
    printf("blockIdx.x : %d, threadIdx.x : %d, offset : %d, value : %d\n", blockIdx.x, 
    threadIdx.x, offset, input[tid]);

}

int main(void){
    
    int dataLen = 16;
    int data[] = {5,2,3,7,4,9,1,6,8,11,10,13,12,15,14,0};
    
    int datasize = sizeof(data);
    printf("%d\n",datasize);

    int* input;
    cudaMalloc((void**)&input, datasize);
    cudaMemcpy(input,data,datasize,cudaMemcpyHostToDevice);

    dim3 block(4);
    dim3 grid(4);

    //indexCalculation_thread<<<grid,block>>>(input);

    indexCalculation_blockgridThread<<<grid,block>>>(input);
    cudaDeviceSynchronize();
    cudaDeviceReset();

    cudaFree(input);
    return 0;
}