#include "device_launch_parameters.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <cstdlib>
#include <time.h>

#define BLOCK_SIZE 32
#define WA 64 
#define HA 64   
#define HC 3     
#define WC 3
#define WB (WA - WC + 1)
#define HB (HA - HC + 1)


void randomInit(float* data, int size)
{
	for (int i = 0; i < size; ++i)
		data[i] = rand() / (float)RAND_MAX;
    
}

int main(int argc, char** argv)
{

    CUresult error;
    CUdevice cuDevice;

    cuInit(0);
    int deviceCount = 0;
    error = cuDeviceGetCount(&deviceCount);
    printf("device count is %d\n",deviceCount);
    error = cuDeviceGet(&cuDevice,0);
    if(error!=CUDA_SUCCESS){
        printf("CUDA error%d\n",error);
    }

    CUcontext cuContext;
    error = cuCtxCreate(&cuContext,0,cuDevice);
    if(error!=CUDA_SUCCESS){
        printf("CUDA create context error%d\n",error);
    }


    CUmodule module;
    CUfunction ConvolutionFunction;
    const char* module_file = "convPTX.ptx";
    const char* kernel_name = "Convolution";

    error = cuModuleLoad(&module, module_file);
    if(error!=CUDA_SUCCESS){
        printf("CUDA module load error %d\n",error);
    }
    error = cuModuleGetFunction(&ConvolutionFunction,module,kernel_name);
    if(error!=CUDA_SUCCESS){
        printf("CUDA get function error %d\n",error);
    }
    //Convolution <<< grid, threads >>>(d_A, d_B, d_C, HA, WA, HB, WB, HC, WC);
    
    

    //conv kernel launch
	srand(2006);
	cudaError_t error_t;
	cudaEvent_t start_G, stop_G;

	cudaEventCreate(&start_G);
	cudaEventCreate(&stop_G);

	unsigned int size_A = WA * HA;
	unsigned int mem_size_A = sizeof(float) * size_A;
	float* h_A = (float*)malloc(mem_size_A);

	unsigned int size_B = WB * HB;
	unsigned int mem_size_B = sizeof(float) * size_B;
	float* h_B = (float*)malloc(mem_size_B);

	unsigned int size_C = WC * HC;
	unsigned int mem_size_C = sizeof(float) * size_C;
	float* h_C = (float*)malloc(mem_size_C);

	randomInit(h_A, size_A);
	randomInit(h_C, size_C);
    //for(int loop = 0; loop < size_A; loop++){
    //    printf("%f ", h_A[loop]);
    //}
	float* d_A;
	float* d_B;
	float* d_C;
    

	error_t = cudaMalloc((void**)&d_A, mem_size_A);
	if (error_t != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s  in cudaMalloc for A\n", cudaGetErrorString(error_t));
		return EXIT_FAILURE;
	}

	error_t = cudaMalloc((void**)&d_B, mem_size_B);//results
	if (error_t != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s  in cudaMalloc for B\n", cudaGetErrorString(error_t));
		return EXIT_FAILURE;
	}

	error_t = cudaMalloc((void**)&d_C, mem_size_C);
	if (error_t != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s  in cudaMalloc for C\n", cudaGetErrorString(error_t));
		return EXIT_FAILURE;
	}


	error_t = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
	if (error_t != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s  in cudaMemcpy for A\n", cudaGetErrorString(error_t));
		return EXIT_FAILURE;
	}

	error_t = cudaMemcpy(d_C, h_C, mem_size_C, cudaMemcpyHostToDevice);
	if (error_t != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s  in cudaMemcpy for C\n", cudaGetErrorString(error_t));
		return EXIT_FAILURE;
	}

	//dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	//dim3 grid((WB - 1) / (BLOCK_SIZE - WC + 1), (WB - 1) / (BLOCK_SIZE - WC + 1));
    int *a = 0;
    int *b = 0;
    int *c = 0;
    int *d = 0;
    int *e = 0;
    int *f = 0;

    int status = cudaMalloc((void**)&a,sizeof(int));
    status = cudaMalloc((void**)&b,sizeof(int));
    status = cudaMalloc((void**)&c,sizeof(int));
    status = cudaMalloc((void**)&d,sizeof(int));
    status = cudaMalloc((void**)&e,sizeof(int));
    status = cudaMalloc((void**)&f,sizeof(int));

    size_t input_HA = HA;
    size_t input_HB = HB;
    size_t input_HC = HC;
    size_t input_WA = WA;
    size_t input_WB = WB;
    size_t input_WC = WC;

    cudaMemcpy(a,&input_HA,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(b,&input_HB,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(c,&input_HC,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d,&input_WA,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(e,&input_WB,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(f,&input_WC,sizeof(int),cudaMemcpyHostToDevice);

    void *kernel_params[] = {(float*)&d_A,(float*)&d_B,(float*)&d_C,(int*)&a,(int*)&d,(int*)&b,(int*)&e,(int*)&c,(int*)&f};
    cuLaunchKernel(ConvolutionFunction,2,2,1,32,32,1,NULL,NULL,kernel_params,0);

	//Convolution <<< grid, threads >>>(d_A, d_B, d_C, HA, WA, HB, WB, HC, WC);

	cudaEventRecord(start_G);

	//Convolution <<< grid, threads >>>(d_A, d_B, d_C, HA, WA, HB, WB, HC, WC);
	error_t = cudaGetLastError();
	if (error_t != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s  in launching kernel\n", cudaGetErrorString(error_t));
		return EXIT_FAILURE;
	}

	error_t = cudaDeviceSynchronize();

	if (error_t != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s  in cudaDeviceSynchronize \n", cudaGetErrorString(error_t));
		return EXIT_FAILURE;
	}

	cudaEventRecord(stop_G);

	cudaEventSynchronize(stop_G);

	error_t = cudaMemcpy(h_B, d_B, mem_size_B, cudaMemcpyDeviceToHost);

	if (error_t != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s  in cudaMemcpy for B\n", cudaGetErrorString(error_t));
		return EXIT_FAILURE;
	}


	float miliseconds = 0;
	cudaEventElapsedTime(&miliseconds, start_G, stop_G);

	printf("\n1Time took to compute matrix A of dimensions %d x %d  on GPU is %f ms \n \n \n", WA, HA, miliseconds);

	for (int i = 0;i < HB;i++)
	{
		for (int j = 0;j < WB;j++)
		{
            int i = 1;
			//printf("%f ", h_B[i*HB + j]);
		}
		//printf("\n");
	}

	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return EXIT_SUCCESS;
}
