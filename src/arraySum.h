#pragma once


float arraySum(float *x, int N) {
  float a = 0;
  for (int i=0; i<N; i++)
    a += x[i];
  return a;
}
