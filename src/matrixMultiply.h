#pragma once
#include "_support.h"


// Calculates matrix product.
void matixMultiply(float *a, float *x, float *y, int XR, int XC, int YC) {
  for (int r=0; r<XR; r++) {
    for (int c=0; c<YC; c++) {
      float s = 0;
      for (int i=0; i<XC; i++)
        s += x[r*XC + i] * y[i*YC + c];
        a[r*YC + c] = s;
    }
  }
}


__global__ void matrixMultiplyKernel(float *a, float *x, float *y, int XR, int XC, int YC) {
  DEFINE(tx, ty, bx, by, BX, BY, GX, GY);
  int r = by*BY + ty;
  int c = bx*BX + tx;

  float s = 0;
  for (int i=0; i<XC; i++)
    s += GET2D(x, r, i, XC) * GET2D(y, i, c, YC);
  GET2D(a, r, c, YC) = s;
}


void matrixMultiplyCuda(float *a, float *x, float *y, int XR, int XC, int YC) {
  size_t A1 = XR * YC * sizeof(float);
  size_t X1 = XR * XC * sizeof(float);
  size_t Y1 = XC * YC * sizeof(float);

  float *aD, *xD, *yD;
  TRY( cudaMalloc(&aD, A1) );
  TRY( cudaMalloc(&xD, X1) );
  TRY( cudaMalloc(&yD, Y1) );

  TRY( cudaMemcpy(xD, x, X1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(yD, y, Y1, cudaMemcpyHostToDevice) );

  dim3 threads(16, 16);
  dim3 blocks(CEILDIV(XR, 16), CEILDIV(YC, 16));
  matrixMultiplyKernel<<<blocks, threads>>>(aD, xD, yD, XR, XC, YC);

  TRY( cudaMemcpy(a, aD, A1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(yD) );
  TRY( cudaFree(xD) );
  TRY( cudaFree(aD) );
}
