#include <cuda_runtime.h>
#include <stdio.h>
extern "C" __global__ void kernel_run(){
    printf("Hello World\n");
}

extern "C" __global__ void charPrint(int *input1, int *input2){
    printf("input 1: %d ~~~ input 2: %d\n",input1,input2);
}

extern "C" __global__ void add(int *sum, int *input1,int *input2){
    *sum = *input1+*input2;
    printf("sum = %d\n",&sum);
}


int main(void){
    dim3 block(1,1);
    dim3 grid(1,1);
    kernel_run<<<block, grid>>>();
    
    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}