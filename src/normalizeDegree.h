#pragma once
#include "DenseDiGraph.h"
#include "degree.h"


// Normalizes weights of graph by out-degree.
void normalizeDegree(DenseDiGraph x) {
  int n = x.order;
  for (int i=0; i<n; i++) {
    int d = degree(x, i);
    if (!d) d = n;
    for (int j=0; j<n; j++)
      if (x.weights[j*n + i] != 0 || d == n) x.weights[j*n + i] = 1/d;
  }
}
