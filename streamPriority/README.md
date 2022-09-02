## Stream Priority

#### __host__ ​cudaError_t cudaDeviceGetStreamPriorityRange ( int* leastPriority, int* greatestPriority ):Returns numerical values that correspond to the least and greatest stream priorities. 

#### __host__ ​cudaError_t cudaStreamCreateWithPriority ( cudaStream_t* pStream, unsigned int  flags, int  priority )： Create an asynchronous stream with the specified priority. 

#### __host__ ​ __device__ ​cudaError_t cudaEventRecord ( cudaEvent_t event, cudaStream_t stream = 0 )：Captures in event the contents of stream at the time of this call. event and stream must be on the same CUDA context. 

#### __host__ ​cudaError_t cudaEventElapsedTime ( float* ms, cudaEvent_t start, cudaEvent_t end ): Computes the elapsed time between events. 
