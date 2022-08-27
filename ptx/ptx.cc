#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>

int main(){
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
    CUfunction function_kernel_1;
    CUfunction function_kernel_2;
    CUfunction function_kernel_3;

    const char* module_file = "ptx.ptx";
    const char* kernel_1_name = "kernel_run";
    const char* kernel_2_name = "charPrint";
    const char* kernel_3_name = "add";

    error = cuModuleLoad(&module, module_file);
    if(error!=CUDA_SUCCESS){
        printf("CUDA module load error %d\n",error);
    }

    error = cuModuleGetFunction(&function_kernel_1,module,kernel_1_name);
    if(error!=CUDA_SUCCESS){
        printf("CUDA get function error %d\n",error);
    }

    //launch kernel 1
    error = cuLaunchKernel(function_kernel_1,1,1,1,1,1,1,NULL,NULL,0,0);
    //printf("%d\n",error);
    cudaThreadSynchronize();
    printf("Run Kernel 1 Success \n\n");

    //launch kernel 2
    error = cuModuleGetFunction(&function_kernel_2,module,kernel_2_name);
    if(error!=CUDA_SUCCESS){
        printf("CUDA get function error %d\n",error);
    }
    int input1 = 20;
    int input2 = 30;
    void *kernel_params[] ={(int*)&input1,(int*)&input2};
    error = cuLaunchKernel(function_kernel_2,1,1,1,1,1,1,NULL,NULL,kernel_params,0);
    //printf("%d\n",error);
    cudaThreadSynchronize();
    printf("Run Kernel 2 Success \n\n");

    //launch kernel 3
    int *a = 0;
    int *b = 0;
    int *c = 0;

    int status = cudaMalloc((void**)&a,sizeof(int));
    status = cudaMalloc((void**)&b,sizeof(int));
    status = cudaMalloc((void**)&c,sizeof(int));

    size_t input_a = 20;
    size_t input_b = 30;

    cudaMemcpy(a,&input_a,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(b,&input_b,sizeof(int),cudaMemcpyHostToDevice);
    void *kernel_params_2[] = {(int*)&c,(int*)&a,(int*)&b};
    error = cuModuleGetFunction(&function_kernel_3,module,kernel_3_name);
    if(error!=CUDA_SUCCESS){
        printf("CUDA get function error %d\n",error);
    }
    error = cuLaunchKernel(function_kernel_3,1,1,1,1,1,1,NULL,NULL,kernel_params_2,0);
    cudaThreadSynchronize();

    int sum_result;
    cudaMemcpy(&sum_result,c,sizeof(int),cudaMemcpyDeviceToHost);
    printf("result on host: %d\n",sum_result);
      printf("Run Kernel 3 Success \n");

    return 1;
}