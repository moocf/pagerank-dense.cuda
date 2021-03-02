PageRank (PR) algorithm for dense graphs using CUDA Toolkit.

> NOTE: Execution time doesnt include memory allocation. It is done ahead of time.

```bash
0/4 tests failed.
Loading graph ...
tcmalloc: large alloc 3365863424 bytes == 0x55ac95456000 @  0x7fb4c2d39887 0x55ac8e10c14d 0x55ac8e10b4c7 0x7fb4c1b39bf7 0x55ac8e10b66a
order: 29008 size: 841464064 {}
[28677.9 ms] pageRank
[00444.6 ms] pageRankCuda


==2739== NVPROF is profiling process 2739, command: ./a.out data/aug2d.mtx
==2739== Profiling application: ./a.out data/aug2d.mtx
==2739== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   61.86%  719.35ms         7  102.76ms  1.3120us  719.34ms  [CUDA memcpy HtoD]
                   38.13%  443.40ms        24  18.475ms  18.043ms  18.573ms  void pageRankKernel<float>(float*, float*, float*, int, float)
                    0.01%  83.231us        24  3.4670us  3.1030us  3.6160us  void errorAbsKernel<float>(float*, float*, float*, int)
                    0.00%  57.440us        29  1.9800us  1.3760us  10.592us  [CUDA memcpy DtoH]
                    0.00%  8.8310us         1  8.8310us  8.8310us  8.8310us  void dotProductKernel<int>(int*, int*, int*, int)
                    0.00%  8.6720us         1  8.6720us  8.6720us  8.6720us  void errorAbsKernel<int>(int*, int*, int*, int)
                    0.00%  8.6070us         1  8.6070us  8.6070us  8.6070us  void sumKernel<int>(int*, int*, int)
                    0.00%  4.5120us         1  4.5120us  4.5120us  4.5120us  void fillKernel<int>(int*, int, int)
                    0.00%  1.6000us         1  1.6000us  1.6000us  1.6000us  void fillKernel<float>(float*, int, float)
      API calls:   84.47%  1.16353s        36  32.320ms  8.0920us  719.50ms  cudaMemcpy
                   15.21%  209.49ms        13  16.115ms  3.5670us  205.78ms  cudaMalloc
                    0.23%  3.1112ms        13  239.32us  3.5800us  2.5906ms  cudaFree
                    0.05%  661.34us        53  12.478us  5.0480us  50.800us  cudaLaunchKernel
                    0.03%  396.95us         1  396.95us  396.95us  396.95us  cuDeviceTotalMem
                    0.01%  157.38us        97  1.6220us     134ns  66.556us  cuDeviceGetAttribute
                    0.00%  37.001us         1  37.001us  37.001us  37.001us  cuDeviceGetName
                    0.00%  7.3980us         1  7.3980us  7.3980us  7.3980us  cuDeviceGetPCIBusId
                    0.00%  1.9710us         3     657ns     134ns  1.4210us  cuDeviceGetCount
                    0.00%  1.4460us         2     723ns     249ns  1.1970us  cuDeviceGet
                    0.00%     230ns         1     230ns     230ns     230ns  cuDeviceGetUuid
```

<br>
<br>


## Usage

```bash
# Download program
rm -rf pagerank-dense
git clone https://github.com/cudaf/pagerank-dense
```

```bash
# Run
cd pagerank-dense && nvcc -Xcompiler -fopenmp -O3 main.cu && nvprof ./a.out data/aug2d.mtx
```

<br>
<br>


## References

- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](http://snap.stanford.edu/class/cs246-videos-2019/lec9_190205-cs246-720.mp4)
- [CUDA by Example :: Jason Sanders, Edward Kandrot](http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf)
- [RAPIDS CUDA DataFrame Internals for C++ Developers - S91043](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s91043-rapids-cuda-dataframe-internals-for-c++-developers.pdf)
