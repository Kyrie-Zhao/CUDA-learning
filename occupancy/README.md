## Occupancy

#### __host__ ​cudaError_t cudaEventCreate ( cudaEvent_t* event ): Creates an event object. 

#### Occupancy is defined in terms of active blocks per multiprocessor

#### template < class T > __host__ ​cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor ( int* numBlocks, T func, int  blockSize, size_t dynamicSMemSize ) [inline]: Returns occupancy for a device function. 