#pragma once
#include "DenseDiGraph.h"




// Finds out-degree of a node.
template <class T>
int degree(DenseDiGraph<T> x, int i) {
  int N = x.order, a = 0;
  for (int j=0; j<N; j++)
    if (x.weight(i, j) != 0) a++;
  return a;
}
