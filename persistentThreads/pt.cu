#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// GTS450 sm_21
#define NUM_SM 34 // no. of streaming multiprocessors
#define NUM_WARP_PER_SM 48 // maximum no. of resident warps per SM
#define NUM_BLOCK_PER_SM 8 // maximum no. of resident blocks per SM
#define NUM_BLOCK NUM_SM * NUM_BLOCK_PER_SM
#define NUM_WARP_PER_BLOCK NUM_WARP_PER_SM / NUM_BLOCK_PER_SM
#define WARP_SIZE 32

#define NUM_TASK 10000

__device__ unsigned int headDev = 0;

__global__ void persistentKernel(int* task)
{
	// warp-wise head index of tasks in a block
	__shared__ volatile unsigned int headBlock[NUM_WARP_PER_BLOCK];

	volatile unsigned int& headWarp = headBlock[threadIdx.y];
	while (true)
	{
		// let lane 0 fetch [wh, wh + WARP_SIZE - 1] for a warp
		if (threadIdx.x == 0) {
			headWarp = atomicAdd(&headDev, WARP_SIZE);
		}
		// task index per thread in a warp
		unsigned int taskIdx = headWarp + threadIdx.x;

		if (taskIdx >= NUM_TASK) {
			return;
		}

		task[taskIdx] += taskIdx;
	} ;
}

int main()
{
	int task[NUM_TASK] = { 0 };
	unsigned int head = 0;

	cudaSetDevice(0);
	int* taskDev = nullptr;
	cudaMalloc((void**)&taskDev, NUM_TASK * sizeof(int));
	cudaMemcpy(taskDev, task, NUM_TASK * sizeof(int), cudaMemcpyHostToDevice);

	persistentKernel << < NUM_BLOCK, dim3(WARP_SIZE, NUM_WARP_PER_BLOCK) >> >(taskDev);
	fprintf(stderr, "Kernel launch detected: %s\n", cudaGetErrorString(cudaGetLastError()));

	cudaDeviceSynchronize();
	cudaMemcpy(task, taskDev, NUM_TASK * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&head, headDev, sizeof(unsigned int));
	printf("head: %d\n", head);

#ifdef _DEBUG
	for (int i = 0; i < NUM_TASK; i++) {
		if (task[i] != i) {
			printf("failed.");
			break;
		}
	}
#endif

	cudaFree(taskDev);
	cudaDeviceReset();

	return 0;
}