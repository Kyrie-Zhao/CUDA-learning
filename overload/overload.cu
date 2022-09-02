#define THREAD_N 256
#define N 1024
#define DIV_UP(a, b) (((a) + (b)-1) / (b))

#include <stdio.h>
#include "overload.cuh"

const char *sampleName = "C++ Overloading";

#define OUTPUT_ATTR(attr)                                         \
  printf("Shared Size:   %d\n", (int)attr.sharedSizeBytes);       \
  printf("Constant Size: %d\n", (int)attr.constSizeBytes);        \
  printf("Local Size:    %d\n", (int)attr.localSizeBytes);        \
  printf("Max Threads Per Block: %d\n", attr.maxThreadsPerBlock); \
  printf("Number of Registers: %d\n", attr.numRegs);              \
  printf("PTX Version: %d\n", attr.ptxVersion);                   \
  printf("Binary Version: %d\n", attr.binaryVersion);

bool check_func1(int *hInput, int *hOutput, int a){
    for (int i = 0; i<N ; ++i){
        int cpuRes = hInput[i] * a + i;
        if (hOutput[i]!=cpuRes){
            return false;
        }
    }
    return true;
}


bool check_func2(int2 *hInput, int *hOutput, int a){
    for (int i = 0; i<N; ++i){
        int cpuRes = (hInput[i].x + hInput[i].y) * a + i;
        if (hOutput[i]!=cpuRes){
            return false;
        }
    }
    return true;
}

bool check_func3(int *hInput1, int *hInput2, int *hOutput, int a){
    for (int i = 0; i<N ; ++i){
        if (hOutput[i]!=(hInput1[i]+hInput2[i])*a + i){
            return false;
        }
    }
    return true;
}

int main(int argc, const char *argv[]){
    int *hInput = NULL;
    int *hOutput = NULL;
    int *dInput = NULL;
    int *dOutput = NULL;

    printf("%s starting \n",sampleName);

    int deviceID = 0;
    cudaSetDevice(deviceID);

    //Allocate memory
    cudaMalloc(&dInput, sizeof(int)*N*2);
    cudaMalloc(&dOutput, sizeof(int)*N);

    cudaMallocHost(&hInput, sizeof(int)*N*2);
    cudaMallocHost(&hOutput, sizeof(int)*N);

    // init
    for (int i =0; i<N*2;i++){
        hInput[i] = i;
    }

    //copy data from host to device
    cudaMemcpy(dInput, hInput, sizeof(int)*N*2, cudaMemcpyHostToDevice);

    //test overloading
    bool testResult = true;
    bool funcResult = true;
    int a = 1;

    void (*func1)(const int *, int *, int);
    void (*func2)(const int2 *, int *, int);
    void (*func3)(const int *, const int *, int *, int);
    
    struct cudaFuncAttributes attr;

    //overload function 1
    func1 = simple_kernel;
    memset(&attr, 0, sizeof(attr));
    cudaFuncSetCacheConfig(*func1, cudaFuncCachePreferShared);
    cudaFuncGetAttributes(&attr, *func1);
    OUTPUT_ATTR(attr);
    
    (*func1)<<<DIV_UP(N,THREAD_N),THREAD_N>>>(dInput,dOutput, a);
    cudaMemcpy(hOutput, dOutput, sizeof(int)*N, cudaMemcpyDeviceToHost);
    funcResult = check_func1(hInput,hOutput,a);
    printf("simple_kernel(const int *pIn, int *pOut, int a) %s\n\n", funcResult ? "PASSED" : "FAILED");
    testResult &= funcResult;

    //overload function 2
    func2 = simple_kernel;
    memset(&attr, 0, sizeof(attr));
    cudaFuncSetCacheConfig(*func2, cudaFuncCachePreferShared);
    cudaFuncGetAttributes(&attr, *func2);
    OUTPUT_ATTR(attr);

    (*func2)<<<DIV_UP(N,THREAD_N),THREAD_N>>>((int2*)dInput,dOutput, a);
    cudaMemcpy(hOutput, dOutput, sizeof(int)*N, cudaMemcpyDeviceToHost);
    funcResult = check_func2(reinterpret_cast<int2 *>(hInput),hOutput,a);
    printf("simple_kernel(const int2 *pIn, int *pOut, int a) %s\n\n", funcResult ? "PASSED" : "FAILED");
    testResult &= funcResult;

    //overload function 3
    func3 = simple_kernel;
    memset(&attr, 0, sizeof(attr));
    cudaFuncSetCacheConfig(*func3, cudaFuncCachePreferShared);
    cudaFuncGetAttributes(&attr, *func3);
    OUTPUT_ATTR(attr);

    (*func3)<<<DIV_UP(N,THREAD_N),THREAD_N>>>(dInput,dInput+N,dOutput, a);
    cudaMemcpy(hOutput, dOutput, sizeof(int)*N, cudaMemcpyDeviceToHost);
    funcResult = check_func3(&hInput[0],&hInput[N],hOutput,a);
    printf("simple_kernel(const int *pIn1, const int *pIn2, int *pOut, int a) %s\n\n", funcResult ? "PASSED" : "FAILED");
    testResult &= funcResult;    

    cudaFree(dInput);
    cudaFree(dOutput);
    cudaFreeHost(hOutput);
    cudaFreeHost(hInput);

    cudaDeviceSynchronize();
    exit(testResult ? EXIT_SUCCESS:EXIT_FAILURE);


}