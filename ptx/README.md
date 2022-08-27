## ptx

### Command: nvcc -ptx ptx.cu & g++ ptx.cc -o ptx -ldl -lpthread -L/usr/local/cuda-10.2/lib64 -lcudart -lcuda -lstdc++ 

#### 将.cu编译成.ptx，然后在源码中加载这个.ptx并使用里面的func，其中.ptx是与平台无关的汇编代码

#### CUresult cuDeviceGetCount ( int* count ): Returns the number of compute-capable devices. 

#### CUresult cuDeviceGet ( CUdevice* device, int  ordinal ): Returns a handle to a compute device.

#### CUDA Context: The context holds all the management data to control and use the device. For instance, it holds the list of allocated memory, the loaded modules that contain device code, the mapping between CPU and GPU memory for zero copy, etc.

#### CUresult cuCtxCreate ( CUcontext* pctx, unsigned int  flags, CUdevice dev ): Create a CUDA context. 

#### Modules are dynamically loadable packages of device code and data, akin to DLLs in Windows, that are output by nvcc (see Compilation with NVCC). The names for all symbols, including functions, global variables, and texture or surface references, are maintained at module scope so that modules written by independent third parties may interoperate in the same CUDA context. 


#### CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name): Returns a function handle Returns in *hfunc the handle of the function of name name located in module hmod. If no function of that name exists, ::cuModuleGetFunction() returns ::CUDA_ERROR_NOT_FOUND.

#### __host__ ​cudaError_t cudaLaunchKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream ): Launches a device function. 

#### cuKernelLaunch的參數設置注意。

#### void * 不確定類型，取值的時候用(int*)

![avatar](http://docs.nvidia.com/cuda/parallel-thread-execution/graphics/memory-hierarchy.png)