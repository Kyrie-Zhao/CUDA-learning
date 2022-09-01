#include <cuda_runtime.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdio>
#include <stdio.h>
#include <cuda.h>
__global__ static void timedReduction(const float *input, float *output,
                                      clock_t *timer) {
  extern __shared__ float shared[];

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  if (tid == 0) timer[bid] = clock();

  // Copy input.
  shared[tid] = input[tid];
  shared[tid + blockDim.x] = input[tid + blockDim.x];

  // Perform reduction to find minimum.
  for (int d = blockDim.x; d > 0; d /= 2) {
    __syncthreads();

    if (tid < d) {
      float f0 = shared[tid];
      float f1 = shared[tid + d];

      if (f1 < f0) {
        shared[tid] = f1;
      }
    }
  }

  // Write result.
  if (tid == 0) output[bid] = shared[0];

  __syncthreads();

  if (tid == 0) timer[bid + gridDim.x] = clock();
}

#define NUM_BLOCKS 64
#define NUM_THREADS 256

int main(int argc, char **argv) {
    float *dinput = NULL;
    float *doutput = NULL;
    clock_t *dtimer = NULL;

    clock_t timer[NUM_BLOCKS*2];
    float input[NUM_THREADS*2];

    for (int i = 0; i < NUM_THREADS * 2; i++) {
        input[i] = (float)i;
    }

    cudaMalloc((void **)&dinput, sizeof(float) * NUM_THREADS * 2);
    cudaMalloc((void **)&doutput, sizeof(float)*NUM_BLOCKS);
    cudaMalloc((void **)&dtimer, sizeof(clock_t) * NUM_BLOCKS * 2);
    
    cudaMemcpy(dinput, input, sizeof(float) * NUM_THREADS * 2, cudaMemcpyHostToDevice);
    timedReduction<<<NUM_BLOCKS, NUM_THREADS, sizeof(float) * 2 * NUM_THREADS>>>(dinput, doutput, dtimer);

    cudaMemcpy(timer, dtimer, sizeof(clock_t) * NUM_BLOCKS * 2,cudaMemcpyDeviceToHost);

    cudaFree(dinput);
    cudaFree(doutput);
    cudaFree(dtimer);

    long double averageElapsedClock = 0;
    for(int t = 0;t<NUM_BLOCKS;t++){
        averageElapsedClock += (long double)(timer[t + NUM_BLOCKS] - timer[t]);
    }

    averageElapsedClock = averageElapsedClock / NUM_BLOCKS;
    printf("Average clocks/block = %Lf\n", averageElapsedClock);

    return EXIT_SUCCESS;
}