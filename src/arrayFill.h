#pragma once


float* arrayFill(float *x, int N, float v) {
  for (int i=0; i<N; i++)
    x[i] = v;
  return x;
}
