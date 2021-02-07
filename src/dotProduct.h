#pragma once
#include "_support.h"

// Constants
#ifdef _THREADS
#undef _THREADS
#endif
#define _THREADS 4


// Finds sum of element-by-element product of 2 vectors (arrays).
float dotProduct(float *x, float *y, int N) {
  float a = 0;
  for (int i=0; i<N; i++)
    a += x[i] * y[i];
  return a;
}


__global__ void dotProductKernel(float *a, float *x, float *y, int N) {
  DEFINE(tx, ty, bx, by, BX, BY, GX, GY);
  UNUSED(ty); UNUSED(by); UNUSED(BY); UNUSED(GY);
  __shared__ float cache[_THREADS];
  int i = bx*BX + tx, t = tx;
  int s = 0;

  for (; i<N; i+=BX*GX)
    s += x[i] * y[i];
  cache[t] = s;

  __syncthreads();
  for (int T=BX/2; T!=0; T/=2) {
    if (t < T) cache[t] += cache[t + T];
    __syncthreads();
  }

  if (t == 0) a[bx] = cache[0];
}


float dotProductCuda(float *x, float *y, int N) {
  int threads = _THREADS;
  int blocks = MAX(CEILDIV(N, threads), 2);
  size_t X1 = N * sizeof(float);
  size_t A1 = blocks * sizeof(float);
  float *aPartial = (float*) malloc(A1);

  float *xD, *yD, *aPartialD;
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

  return arraySum(aPartial, blocks);
}
