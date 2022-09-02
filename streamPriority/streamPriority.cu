#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <time.h>

#define TOTAL_SIZE 256*1024*1024
#define EACH_SIZE 128*1024*1024

// threadblocks
#define TBLOCKS 1024
#define THREADS 512

// Error Throw on equality
#define TBLOCKS 1024
#define THREADS  512

// copy from source -> destination arrays
__global__ void memcpy_kernel(int *dst, int *src, size_t n)
{
    int num = gridDim.x * blockDim.x; //threads in total
    int id = blockDim.x * blockIdx.x + threadIdx.x; //offset+threadid.x

    for (int i = id; i < n / sizeof(int); i += num)
    {
        dst[i] = src[i];
    }
}

// initialise memory
void mem_init(int *buf, size_t n) {
  for (int i = 0; i < n / sizeof(int); i++) {
    buf[i] = i;
  }
}

int main(int argc, char **argv) {

    // get the range of priorities available
    // [ greatest_priority, lowest_priority ]
    int *priority_low;
    int *priority_hi;
    //printf("elapsed time of kernels launched to LOW priority stream: %.3lf ms\n",ms_low);
    //printf("elapsed time of kernels launched to HI  priority stream: %.3lf ms\n",ms_hi);

    priority_low = (int *)malloc(sizeof(int));
    priority_hi = (int *)malloc(sizeof(int));
    cudaDeviceGetStreamPriorityRange(priority_low, priority_hi);

    printf("CUDA stream priority range: LOW: %d to HIGH: %d\n", *priority_low,*priority_hi);

    cudaStream_t st_low;
    cudaStream_t st_hi;
    
    cudaStreamCreateWithPriority(&st_low, cudaStreamNonBlocking,*priority_low);
    cudaStreamCreateWithPriority(&st_hi, cudaStreamNonBlocking, *priority_hi);

    size_t size;
    size = TOTAL_SIZE;
    
    // initialise host data
    int *h_src_low;
    int *h_src_hi;
    h_src_low = (int *)malloc(size);
    h_src_hi = (int *)malloc(size);
    mem_init(h_src_low, size);
    mem_init(h_src_hi, size);

    // initialise device data
    int *h_dst_low;
    int *h_dst_hi;
    h_dst_low = (int *)malloc(size);
    h_dst_hi = (int *)malloc(size);
    memset(h_dst_low, 0, size);
    memset(h_dst_hi, 0, size);

     // copy source data -> device
    int *d_src_low;
    int *d_src_hi;
    cudaMalloc(&d_src_low, size);
    cudaMalloc(&d_src_hi, size);
    cudaMemcpy(d_src_low, h_src_low, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_src_hi, h_src_hi, size, cudaMemcpyHostToDevice);

    // allocate memory for memcopy destination
    int *d_dst_low;
    int *d_dst_hi;
    cudaMalloc(&d_dst_low, size);
    cudaMalloc(&d_dst_hi, size);

    // create some events
    cudaEvent_t ev_start_low;
    cudaEvent_t ev_start_hi;
    cudaEvent_t ev_end_low;
    cudaEvent_t ev_end_hi;
    cudaEventCreate(&ev_start_low);
    cudaEventCreate(&ev_start_hi);
    cudaEventCreate(&ev_end_low);
    cudaEventCreate(&ev_end_hi);

    // call pair of kernels repeatedly (with different priority streams)
    cudaEventRecord(ev_start_low, st_low);
    cudaEventRecord(ev_start_hi, st_hi);

    for (int i = 0; i < TOTAL_SIZE; i += EACH_SIZE) {
        printf("Memcpy Kernel\n");
        int j = i / sizeof(int);
        memcpy_kernel<<<TBLOCKS, THREADS, 0, st_low>>>(d_dst_low + j, d_src_low + j,EACH_SIZE);
        memcpy_kernel<<<TBLOCKS, THREADS, 0, st_hi>>>(d_dst_hi + j, d_src_hi + j,EACH_SIZE);
    }
     

    cudaEventRecord(ev_end_low, st_low);
    cudaEventRecord(ev_end_hi, st_hi);

    cudaEventSynchronize(ev_end_low);
    cudaEventSynchronize(ev_end_hi);


    size = TOTAL_SIZE;
    cudaMemcpy(h_dst_low, d_dst_low, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dst_hi, d_dst_hi, size, cudaMemcpyDeviceToHost);

    // check results of kernels
    memcmp(h_dst_low, h_src_low, size);
    memcmp(h_dst_hi, h_src_hi, size);

    // check timings
    float ms_low;
    float ms_hi;
    cudaEventElapsedTime(&ms_low, ev_start_low, ev_end_low);
    cudaEventElapsedTime(&ms_hi, ev_start_hi, ev_end_hi);

    printf("elapsed time of kernels launched to LOW priority stream: %.3lf ms\n", ms_low);
    printf("elapsed time of kernels launched to HI  priority stream: %.3lf ms\n",ms_hi);
    exit(EXIT_SUCCESS);
}