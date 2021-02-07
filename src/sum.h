#pragma once
#include <array>
#include <vector>

using namespace std;




template <class T>
T sum(T *x, int N) {
  T a = T();
  for (int i=0; i<N; i++)
    a += x[i];
  return a;
}


template <class T, size_t N>
T sum(array<T, N>& x) {
  T a = T();
  for (auto& v : x)
    a += v;
  return a;
}


template <class T>
T sum(vector<T>& x) {
  T a = T();
  for (auto& v : x)
    a += v;
  return a;
}


template <class T>
__device__ T sumBlock(T *a, int N) {
  int t = threadIdx.x;
  int B = blockDim.x;
  for (int T=B/2; T>0; T/=2) {
    if (t < T) a[t] += a[T+t];
    __syncthreads();
  }
  return a[0];
}

template <class T>
__device__ T sumBlocks(T *a, T *x, int N) {
  T s = T();
  for (int i=t; i<N; i+=B)
    s += x[i];

  __syncthreads();

}


template <class T>
__global__ void sumGrid(T *a, T *x, int N) {
  DEFINE(t, ty, b, by, B, BY, G, GY);
  __shared__ T cache[_THREADS];
  for (int i=B*b+t; i<N; i+=BX*GX)

  if (t == 0) a[b] = cache[0];
}


template <class T>
T sumCuda(T *x, int N) {
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
