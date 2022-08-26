#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
__global__ void sumarray(int* a, int* b, int* c, int size){
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
      c[gid] = a[gid] + b[gid];
    }   
}

void sum_array_cpu(int* a, int* b, int* c, int size)
{
    for ( int i = 0;i < size;i++ ) {
        c[i] = a[i] + b[i];      
    }
}

void compare_arrays(int* a, int* b, int size)
{
    for ( int i = 0;i < size;i++ ) {
      if (a[i] != b[i]) { 
          printf("Arrays are different!");
          return;
      }
    }
    printf("Arrays are same\n");
}

int main(void){
    int size = 10000;
    int block_size = 128;
    int nBytes = size * sizeof(int);

    int* h_a, *h_b, *gpu_results, *h_c;

    h_a = (int*)malloc(nBytes);
    h_b = (int*)malloc(nBytes);
    h_c = (int*)malloc(nBytes);
    gpu_results = (int*)malloc(nBytes);

    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0;i < size;i++) {
        h_a[i] = (int)(rand() & 0xff);
        h_b[i] = (int)(rand() & 0xff);
    }
    
    sum_array_cpu(h_a, h_b, h_c, size);
    //printf("%d",&h_a[1]);
    memset(gpu_results, 0, nBytes);


    int* d_a, *d_b, *d_c;
    cudaMalloc((int**)&d_a, nBytes);
    cudaMalloc((int**)&d_b, nBytes);
    cudaMalloc((int**)&d_c, nBytes);

    cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);


    // launching the grid
    dim3 block(block_size);
    dim3 grid((size / block.x) + 1);

    sumarray<<<grid, block>>> (d_a, d_b, d_c, size);
    cudaDeviceSynchronize();


    cudaMemcpy(gpu_results, d_c, nBytes, cudaMemcpyDeviceToHost);


    // array comparison
    compare_arrays(h_c, gpu_results, size);



    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    

    free(gpu_results);
    cudaDeviceReset();


    return 0;

}