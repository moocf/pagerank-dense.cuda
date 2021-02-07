#pragma once
#include <array>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include "_support.h"
#include "ceilDiv.h"
#include "sum.h"

using namespace std;

// Constants
#ifndef _THREADS
#define _THREADS 64
#endif


// Finds sum of element-by-element product of 2 vectors (arrays).
template <class T>
T dotProduct(T *x, T *y, int N) {
  T a = T();
  for (int i=0; i<N; i++)
    a += x[i] * y[i];
  return a;
}


template <class T, size_t N>
T dotProduct(array<T, N>& x, array<T, N>& y) {
  T a = T();
  for (size_t i=0; i<N; i++)
    a += x[i] * y[i];
  return a;
}


template <class T>
T dotProduct(vector<T>& x, vector<T>& y) {
  T a = T();
  for (size_t i=0, I=x.size(); i<I; i++)
    a += x[i] * y[i];
  return a;
}



template <class T>
__global__ void dotProductKernel(T *a, T *x, T *y, int N) {
  DEFINE(tx, ty, bx, by, BX, BY, GX, GY);
  UNUSED(ty); UNUSED(by); UNUSED(BY); UNUSED(GY);
  __shared__ T cache[_THREADS];
  int i = bx*BX + tx;
  T s = T();

  for (; i<N; i+=BX*GX)
    s += x[i] * y[i];
  cache[tx] = s;

  __syncthreads();
  sumReduce(cache, _THREADS, tx);
  if (tx == 0) a[bx] = cache[0];
}


template <class T>
__device__ void sumBlock(T *a, T *x, int N, int i, int DI) {

  for (; i<N; i+=DI)

}

template <class T>
__device__ void sumReduce(T* a, int N, int i) {
  for (N=N/2; N>0; N/=2) {
    if (i < N) a[i] += a[N+i];
    __syncthreads();
  }
}



template <class T>
__device__ T dotProductKernel(T *x, T *y, int N) {
  DEFINE(tx, ty, bx, by, BX, BY, GX, GY);
  __shared__ T cache[_THREADS];
  __shared__ T total;

  T s = T();
  for (int i=tx; i<N; i+=BX)
    s += x[i] * y[i];
  cache[tx] = s;

  __syncthreads();

}


template <class T>
T dotProductCuda(T *x, T *y, int N) {
  int threads = _THREADS;
  int blocks = max(ceilDiv(N, threads), 2);
  size_t X1 = N * sizeof(T);
  size_t A1 = blocks * sizeof(T);
  T *aPartial = (T*) malloc(A1);

  T *xD, *yD, *aPartialD;
  TRY( cudaMalloc(&xD, X1) );
  TRY( cudaMalloc(&yD, X1) );
  TRY( cudaMalloc(&aPartialD, A1) );
  TRY( cudaMemcpy(xD, x, X1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(yD, y, X1, cudaMemcpyHostToDevice) );

  dotProductKernel<<<blocks, threads>>>(aPartialD, xD, yD, N);

  TRY( cudaMemcpy(aPartial, aPartialD, A1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(yD) );
  TRY( cudaFree(xD) );
  TRY( cudaFree(aPartialD) );

  return sum(aPartial, blocks);
}


template <class T, size_t N>
T dotProductCuda(array<T, N>& x, array<T, N>& y) {
  return dotProductCuda(x.data(), y.data(), N);
}


template <class T>
T dotProductCuda(vector<T>& x, vector<T>& y) {
  return dotProductCuda(x.data(), y.data(), x.size());
}
