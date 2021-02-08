#pragma once
#include "_support.h"
#include "ceilDiv.h"


// Calculates matrix product.
template <class T>
void matixMultiply(T *a, T *x, T *y, int XR, int XC, int YC) {
  for (int r=0; r<XR; r++) {
    for (int c=0; c<YC; c++) {
      T s = T();
      for (int i=0; i<XC; i++)
        s += x[r*XC + i] * y[i*YC + c];
        a[r*YC + c] = s;
    }
  }
}



template <class T>
__global__ void matrixMultiplyKernel(T *a, T *x, T *y, int XR, int XC, int YC) {
  DEFINE2D(tx, ty, bx, by, BX, BY, GX, GY);
  UNUSED(GX, GY);
  int r = by*BY + ty;
  int c = bx*BX + tx;

  T s = T();
  for (int i=0; i<XC; i++)
    s += GET2D(x, r, i, XC) * GET2D(y, i, c, YC);
  GET2D(a, r, c, YC) = s;
}


template <class T>
void matrixMultiplyCuda(T *a, T *x, T *y, int XR, int XC, int YC) {
  size_t A1 = XR * YC * sizeof(T);
  size_t X1 = XR * XC * sizeof(T);
  size_t Y1 = XC * YC * sizeof(T);

  T *aD, *xD, *yD;
  TRY( cudaMalloc(&aD, A1) );
  TRY( cudaMalloc(&xD, X1) );
  TRY( cudaMalloc(&yD, Y1) );

  TRY( cudaMemcpy(xD, x, X1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(yD, y, Y1, cudaMemcpyHostToDevice) );

  dim3 threads(16, 16);
  dim3 blocks(ceilDiv(XR, 16), ceilDiv(YC, 16));
  matrixMultiplyKernel<<<blocks, threads>>>(aD, xD, yD, XR, XC, YC);

  TRY( cudaMemcpy(a, aD, A1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(yD) );
  TRY( cudaFree(xD) );
  TRY( cudaFree(aD) );
}
