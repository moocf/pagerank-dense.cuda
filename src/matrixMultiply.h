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


__global__ void matrixMultiplyCuda(float *a, float *x, float *y, int XR, int XC, int YC) {
  DEFINE(tx, ty, bx, by, BX, BY, GX, GY);
  int r = by*BY + ty;
  int c = bx*BX + tx;

  float s = 0;
  for (int i=0; i<XC; i++)
    s += GET2D(x, r, i, XC) * GET2D(y, i, c, YC);
  GET2D(a, r, c, YC) = s;
}
