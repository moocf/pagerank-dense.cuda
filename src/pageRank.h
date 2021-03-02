#pragma once
#include <utility>
#include <cmath>
#include <string.h>
#include <omp.h>
#include "_cuda.h"
#include "DenseDiGraph.h"
#include "measureDuration.h"
#include "dotProduct.h"
#include "errorAbs.h"
#include "fill.h"
#include "sum.h"

using namespace std;




// Finds rank of nodes in graph.
template <class T>
struct pageRankOptions {
  T damping;
  T convergence;

  pageRankOptions(T _damping=0.85, T _convergence=1e-6) {
    damping = _damping;
    convergence = _convergence;
  }
};




template <class T>
auto& pageRankCore(vector<T>& a, vector<T>& r, T *w, int N, T p, T E) {
  T e0 = T();
  fill(r, T(1)/N);
  while (1) {
    for (int j=0; j<N; j++) {
      T wjr = dotProduct(&w[N*j], r.data(), N);
      a[j] = p*wjr + (1-p)/N;
    }
    T e = errorAbs(a, r);
    if (e < E || e == e0) break;
    swap(a, r);
    e0 = e;
  }
  return a;
}


template <class T>
auto pageRank(float& t, T *w, int N, T p, T E) {
  vector<T> r(N);
  vector<T> a(N);
  t = measureDuration([&]() { pageRankCore(a, r, w, N, p, E); });
  return a;
}

template <class T>
auto pageRank(float& t, T *w, int N, pageRankOptions<T> o=pageRankOptions<T>()) {
  return pageRank(t, w, N, o.damping, o.convergence);
}

template <class T>
auto pageRank(float& t, DenseDiGraph<T>& x, pageRankOptions<T> o=pageRankOptions<T>()) {
  return pageRank(t, x.edgeData(), x.order(), o);
}




template <class T>
__global__ void pageRankKernel(T *a, T *r, T *w, int N, T p) {
  DEFINE(t, b, B, G);
  __shared__ T cache[_THREADS];

  for (int j=b; j<N; j+=G) {
    cache[t] = dotProductKernelLoop(&w[N*j], r, N, t, B);
    sumKernelReduce(cache, B, t);
    if (t != 0) continue;
    T wjr = cache[0];
    a[j] = p*wjr + (1-p)/N;
  }
}


template <class T>
T* pageRankCudaCore(T *e, T *a, T *r, T *w, int N, T p, T E) {
  int threads = _THREADS;
  int blocks = min(ceilDiv(N, threads), _BLOCKS);
  int E1 = blocks * sizeof(T);
  T eH[_BLOCKS], e0 = T();
  fillKernel<<<blocks, threads>>>(r, N, T(1)/N);
  while (1) {
    pageRankKernel<<<blocks, threads>>>(a, r, w, N, p);
    errorAbsKernel<<<blocks, threads>>>(e, a, r, N);
    TRY( cudaMemcpy(eH, e, E1, cudaMemcpyDeviceToHost) );
    T f = sum(eH, blocks);
    if (f < E || f == e0) break;
    swap(a, r);
    e0 = f;
  }
  return a;
}

template <class T>
auto pageRankCuda(float& t, T *w, int N, T p, T E) {
  int threads = _THREADS;
  int blocks = min(ceilDiv(N, threads), 1024);
  size_t E1 = blocks * sizeof(T);
  size_t A1 = N * sizeof(T);
  size_t W1 = N*N * sizeof(T);
  vector<T> a(N);

  T *eD, *aD, *bD, *rD, *wD;
  TRY( cudaMalloc(&wD, W1) );
  TRY( cudaMalloc(&aD, A1) );
  TRY( cudaMalloc(&rD, A1) );
  TRY( cudaMalloc(&eD, E1) );
  TRY( cudaMemcpy(wD, w, W1, cudaMemcpyHostToDevice) );

  t = measureDuration([&]() { bD = pageRankCudaCore(eD, aD, rD, wD, N, p, E); });
  TRY( cudaMemcpy(a.data(), bD, A1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(eD) );
  TRY( cudaFree(aD) );
  TRY( cudaFree(rD) );
  TRY( cudaFree(wD) );
  return a;
}


template <class T>
auto pageRankCuda(float& t, T *w, int N, pageRankOptions<T> o=pageRankOptions<T>()) {
  return pageRankCuda(t, w, N, o.damping, o.convergence);
}


template <class T>
auto pageRankCuda(float& t, DenseDiGraph<T>& x, pageRankOptions<T> o=pageRankOptions<T>()) {
  return pageRankCuda(t, x.edgeData(), x.order(), o);
}
