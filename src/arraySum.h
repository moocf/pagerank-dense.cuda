#pragma once


float arraySum(float *x, float *y, int N) {
  float a = 0;
  for (int i=0; i<N; i++)
    a += x[i] * y[i];
  return a;
}
