## Overload
#### int2 These are vector types derived from the basic integer and floating-point types. They are structures and the 1st, 2nd, 3rd, and 4th components are accessible through the fields x, y, z, and w, respectively. They all come with a constructor function of the form make_; for example int2 equals struct{int x, int y,}

#### For shared memory to be useful, you must use data transferred to shared memory several times, using good access patterns, to have it help. 

#### The __shared__ memory space specifier, optionally used together with __device__, declares a variable that: 
- Resides in the shared memory space of a thread block,
- Has the lifetime of the block,
- Has a distinct object per block,
- Is only accessible from all the threads within the block,
- Does not have a constant address.

#### A host thread can set the device it operates on at any time by calling cudaSetDevice(). Device memory allocations and kernel launches are made on the currently set device; streams and events are created in association with the currently set device. If no call to cudaSetDevice() is made, the current device is device 0. 

#### cudaFuncAttributes Struct Reference [Data types used by CUDA Runtime] CUDA function attributes

#### __host__ ​cudaError_t cudaFuncSetCacheConfig ( T* func, cudaFuncCache cacheConfig )： [inline] [C++ API] Sets the preferred cache configuration for a device function 

#### __host__ ​ __device__ ​cudaError_t cudaFuncGetAttributes ( cudaFuncAttributes* attr, const void* func )：Find out attributes for a given function. 