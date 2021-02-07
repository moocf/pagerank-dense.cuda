#pragma once
#include "DenseDiGraph.h"


// Finds out-degree of a node.
template <class T>
int degree(DenseDiGraph<T> x, int i) {
  int n = x.order, a = 0;
  for (int j=0; j<n; j++)
    if (x.weights[j*n + i] != 0) a++;
  return a;
}
