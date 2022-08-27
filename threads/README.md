## Threads, blocks, grids learning

#### cuda_kernel<<<grid_size, block_size, 0, stream>>>(...)

#### 一个 Cuda 程序就是一个 Grid，一块 GPU 就是一组 SM 处理器。程序的提交执行过程，就是如何把 Block 分派到 SM 处理器。Block 不可以分割，SM 处理器也不能分割。一个 SM 处理器可以同时处理多个 Block，这种情况下 Block 需要在其 SM 处理器排队等待调度。

#### <<<Dg, Db>>> 分别代表grid的dimension（里面有多少block）和block的dimension（有多少thread）。这两个参数可以是int 或者CUDA的数据类型dim3

#### 如果需要计算N*N的矩阵，我们就需要(N/blockDim.x,N/blockDim.y) 个block

#### 每个block的运行是相互独立的，这就意味着它们可能是并行，可能是串行。程序（小绿）被分成了8个block（小橙），丢给GPU（小蓝）去运行。有的小蓝（左边）只有两个核心，所以一次最多同时处理两个block，那就要分四批去处理。但是有的小蓝（右边）天赋异禀有四个核心，每次同时跑4个block，就只需要两批处理，比左边的小蓝快了一倍。


![avatar](https://pic4.zhimg.com/80/v2-41fb95f59f9735b04fa1431c3907d13b_720w.jpg)