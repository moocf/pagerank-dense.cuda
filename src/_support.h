#pragma once
#include <stdio.h>
#include <cuda_runtime.h>




// Constants
#ifndef _THREADS
#define _THREADS 64
#endif

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
#define DEFINE_CUDA(t, b, B, G) \
  int t = threadIdx.x; \
  int b = blockIdx.x; \
  int B = blockDim.x; \
  int G = gridDim.x;
#define DEFINE_CUDA2D(tx, ty, bx, by, BX, BY, GX, GY) \
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
#define DEFINE(t, b, B, G) \
  DEFINE_CUDA(t, b, B, G)
#define DEFINE2D(tx, ty, bx, by, BX, BY, GX, GY) \
  DEFINE_CUDA2D(tx, ty, bx, by, BX, BY, GX, GY)
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
#define EMA_GET0(V, ...) V
#define EMA_GET9(_0, _1, _2, _3, _4, _5, _6, _7, _8, V, ...) V
#define _UVC(a) unreferencedVariableCuda(a)
#define UNUSED_CUDA0(...) \
  {}
#define UNUSED_CUDA1(a, ...) \
  { _UVC(a); }
#define UNUSED_CUDA2(a, b, ...) \
  { _UVC(a); _UVC(b); }
#define UNUSED_CUDA3(a, b, c, ...) \
  { _UVC(a); _UVC(b); _UVC(c); }
#define UNUSED_CUDA4(a, b, c, d, ...) \
  { _UVC(a); _UVC(b); _UVC(c); _UVC(d); }
#define UNUSED_CUDA5(a, b, c, d, e, ...) \
  { _UVC(a); _UVC(b); _UVC(c); _UVC(d); _UVC(e); }
#define UNUSED_CUDA6(a, b, c, d, e, f, ...) \
  { _UVC(a); _UVC(b); _UVC(c); _UVC(d); _UVC(e); _UVC(f); }
#define UNUSED_CUDA7(a, b, c, d, e, f, g, ...) \
  { _UVC(a); _UVC(b); _UVC(c); _UVC(d); _UVC(e); _UVC(f); _UVC(g); }
#define UNUSED_CUDA8(a, b, c, d, e, f, g, h, ...) \
  { _UVC(e); _UVC(e); _UVC(e); _UVC(e); _UVC(e); _UVC(e); _UVC(e); _UVC(e); }
#define UNUSED_CUDA(...) \
  EMA_GET0(EMA_GET9(0, ##__VA_ARGS__, \
    UNUSED_CUDA8, UNUSED_CUDA7, UNUSED_CUDA6, UNUSED_CUDA5, \
    UNUSED_CUDA4, UNUSED_CUDA3, UNUSED_CUDA2, UNUSED_CUDA1, UNUSED_CUDA0)(__VA_ARGS__))
#endif
template <class T>
__device__ void unreferencedVariableCuda(T&&) {}

#ifndef UNUSED
#define UNUSED UNUSED_CUDA
#endif



#ifndef GET2D
// Gets value at given row, column of 2D array
#define GET2D(x, r, c, C) (x)[(C)*(r) + (c)]
#endif



#ifndef UINT
typedef unsigned int uint;
#define UINT uint
#endif

#ifndef UINT8
typedef unsigned char uint8;
#define UINT8 uint8
#endif
