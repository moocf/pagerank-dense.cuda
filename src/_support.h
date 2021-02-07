#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>



#ifndef TRY_CUDA
inline void try_cuda(cudaError err, const char* exp, const char* func, int line, const char* file) {
    if (err == cudaSuccess) return;
    fprintf(stderr,
        "%s: %s\n"
        "  in expression %s\n"
        "  at %s:%d in %s\n",
        cudaGetErrorName(err), cudaGetErrorString(err),
        exp,
        func, line, file);
    exit(err);
}

// Prints an error message and exits, if CUDA expression fails.
// TRY_CUDA( cudaDeviceSynchronize() );
#define TRY_CUDA(exp) try_cuda(exp, #exp, __func__, __LINE__, __FILE__)
#endif

#ifndef TRY
#define TRY(exp) TRY_CUDA(exp)
#endif



#ifndef DEFINE_CUDA
// Defines short names for the following variables:
// - threadIdx.x,y
// - blockIdx.x,y
// - blockDim.x,y
// - gridDim.x,y
#define DEFINE_CUDA(tx, ty, bx, by, BX, BY, GX, GY) \
  int tx = threadIdx.x; \
  int ty = threadIdx.y; \
  int bx = blockIdx.x; \
  int by = blockIdx.y; \
  int BX = blockDim.x; \
  int BY = blockDim.y; \
  int GX = gridDim.x;  \
  int GY = gridDim.y;
#endif

#ifndef DEFINE
#define DEFINE(tx, ty, bx, by, BX, BY, GX, GY) \
  DEFINE_CUDA(tx, ty, bx, by, BX, BY, GX, GY)
#endif



#ifndef __SYNCTHREADS
void __syncthreads();
#define __SYNCTHREADS() __syncthreads()
#endif

#ifndef __global__
#define __global__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __shared__
#define __shared__
#endif



#ifndef UNUSED_CUDA
#define UNUSED_CUDA(e) unreferencedVariableCuda(e)
#endif
template <class T>
__device__ void unreferencedVariableCuda(T&&) {}

#ifndef UNUSED
#define UNUSED(e) UNUSED_CUDA(e)
#endif



#ifndef GET2D
// Gets value at given row, column of 2D array
#define GET2D(x, r, c, C) (x)[(r)*(C) + (c)]
#endif



#ifndef UINT
typedef unsigned int uint;
#define UINT uint
#endif

#ifndef UINT8
typedef unsigned char uint8;
#define UINT8 uint8
#endif
