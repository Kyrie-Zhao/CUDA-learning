## Clock

## This example shows how to use the clock function to measure the performance of block of threads of a kernel accurately.

#### "extern" The reason to use dynamically allocated shared memory (as opposed to statically allocated) is similar to one reason why you might want to allocate anything dynamically instead of statically: at compile-time, you don't know the size of the allocation you will want.

####  __syncthreads() __syncthreads()是cuda的内建函数，用于块内线程通信. _syncthreads() is you garden variety thread barrier. Any thread reaching the barrier waits until all of the other threads in that block also reach it. It is designed for avoiding race conditions when loading shared memory, and the compiler will not move memory reads/writes around a __syncthreads(). 其中，最重要的理解是那些可以到达__syncthreads()的线程需要其他可以到达该点的线程，而不是等待块内所有其他线程。 https://www.nvidia.com/content/GTC/documents/SC09_Feng.pdf   

![avatar](https://i.stack.imgur.com/D5gnV.png)
