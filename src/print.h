#pragma once
#include "_support.h"
#include <stdio.h>


// Prints 1D array.
void print(float *x, int N) {
  printf("{");
  for (int i=0; i<N-1; i++)
    printf("%.1f, ", x[i]);
  if (N>0) printf("%.1f", x[N-1]);
  printf("}");
}


// Prints 2D array.
void print(float *x, int R, int C) {
  printf("{\n");
  for (int r=0; r<R; r++) {
    for (int c=0; c<C; c++)
      printf("%.1f, ", GET2D(x, r, c, C));
    printf("\n");
  }
  printf("}\n");
}
