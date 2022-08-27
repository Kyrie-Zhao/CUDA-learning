## Index Calculation

#### __host__ ​ __device__ ​cudaError_t cudaMalloc ( void** devPtr, size_t size ): Allocate memory on the device. 

#### __host__ ​cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind ): Copies data between host and device. 

#### 
blockIdx.x : 3, threadIdx.x : 0, offset : 12, value : 12  

blockIdx.x : 3, threadIdx.x : 1, offset : 12, value : 15  

blockIdx.x : 3, threadIdx.x : 2, offset : 12, value : 14  

blockIdx.x : 3, threadIdx.x : 3, offset : 12, value : 0  

blockIdx.x : 0, threadIdx.x : 0, offset : 0, value : 5  

blockIdx.x : 0, threadIdx.x : 1, offset : 0, value : 2  

blockIdx.x : 0, threadIdx.x : 2, offset : 0, value : 3  

blockIdx.x : 0, threadIdx.x : 3, offset : 0, value : 7  

blockIdx.x : 2, threadIdx.x : 0, offset : 8, value : 8  

blockIdx.x : 2, threadIdx.x : 1, offset : 8, value : 11  

blockIdx.x : 2, threadIdx.x : 2, offset : 8, value : 10  

blockIdx.x : 2, threadIdx.x : 3, offset : 8, value : 13  

blockIdx.x : 1, threadIdx.x : 0, offset : 4, value : 4  

blockIdx.x : 1, threadIdx.x : 1, offset : 4, value : 9  

blockIdx.x : 1, threadIdx.x : 2, offset : 4, value : 1  

blockIdx.x : 1, threadIdx.x : 3, offset : 4, value : 6  



#### block速度不一样
