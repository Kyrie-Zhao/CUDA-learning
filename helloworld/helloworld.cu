#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdio>

__global__ void hello_cuda(){
    printf("hello world cuda\n");
    //device info

}

//device codes wrong,"std::" cannot be directly taken in a device function

/*
__global__ void sys_info(){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for(size_t i = 0; i<deviceCount ; i++){
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp,i);
        std::cout<<"GPU Device: "<<devProp.name<<std::endl;
        std::cout<<"L2 Cache Size: "<<devProp.l2CacheSize<<std::endl;
        std::cout<<"Warp Size: "<<devProp.warpSize<<std::endl;
    }
    
    printf("end");
}
*/

int main(void){
    dim3 block(1,2);
    dim3 grid(1,2);
    hello_cuda<<<block, grid>>>();

    //cudaDeviceSynchronize();
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for(size_t i = 0; i<deviceCount ; i++){
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp,i);
        std::cout<<"GPU Device: "<<devProp.name<<std::endl;
        std::cout<<"L2 Cache Size: "<<devProp.l2CacheSize<<std::endl;
        std::cout<<"Warp Size: "<<devProp.warpSize<<std::endl;
        std::cout<<"Device can possibly execute multiple kernels concurrently: "<<devProp.concurrentKernels<<std::endl;
        std::cout<<"Major compute capability: "<<devProp.major<<std::endl;
        std::cout<<"Number of multiprocessors on device: "<<devProp.multiProcessorCount<<std::endl;
        std::cout<<"Shared memory available per block in bytes: "<<devProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Maximum number of threads per block: "<<devProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Global memory bus width in bits: "<<devProp.memoryBusWidth<<std::endl;


    }
    
    printf("end");
    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}