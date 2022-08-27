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

    cudaError_t e;
    e = cudaLaunchKernel(function_kernel_1,1,1,NULL,0,0);
    printf("%d\n",e);
    cudaThreadSynchronize();


    return 1;
}