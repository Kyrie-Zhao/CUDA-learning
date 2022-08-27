#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>

extern "C" __global__ void kernel_run(){
    printf("Hello World");
}

extern "C" __global__ void charPrint(char *input1, char *input2){
    printf("input 1: %c ==  input 2: %c",input1,input2);
}

extern "C" __global__ void add(int *sum, int *input1, int *input2){
    *sum = *input1 + *input2;
}


int main(void){
    dim3 block(1,1);
    dim3 grid(1,1);
    kernel_run<<<block, grid>>>();
    
    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}