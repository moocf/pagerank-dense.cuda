#pragma once
#include "DenseDiGraph.h"
#include "degree.h"


// Normalizes weights of graph by out-degree.
template <class T>
void normalizeDegree(DenseDiGraph<T>& x) {
  int N = x.order;
  for (int i=0; i<N; i++) {
    int d = degree(x, i);
    if (!d) d = N;
    for (int j=0; j<N; j++)
      if (x.weights[N*j + i] != 0 || d == N) x.weights[N*j + i] = 1/d;
  }
}
