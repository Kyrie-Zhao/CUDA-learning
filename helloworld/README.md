# "Hello CUDA" and some device info

### __host__ ​ __device__ ​cudaError_t cudaDeviceSynchronize ( void ): Wait for compute device to finish. 

### __host__ ​cudaError_t cudaDeviceReset ( void ) Destroy all allocations and reset all state on the current device in the current process. 

### cudaDeviceProp CUDA device properties 

### dim3 is an integer vector type based on uint3 that is used to specify dimensions. 

### A group of threads is called a CUDA block. CUDA blocks are grouped into a grid. A kernel is executed as a grid of blocks of threads (Figure 2). Each CUDA block is executed by one streaming multiprocessor (SM) and cannot be migrated to other SMs in GPU (except during preemption, debugging, or CUDA dynamic parallelism).

### 如果cudaDeviceSynchronize()没有写在Kernel后面就不执行？