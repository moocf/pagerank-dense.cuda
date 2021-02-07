#pragma once
#include "_support.h"
#include "DenseDiGraph.h"


// Finds out-degree of a node.
int degree(DenseDiGraph x, int i) {
  int n = x.order, a = 0;
  for (int j=0; j<n; j++)
    if (x.weights[j*n + i] !== 0) a++;
  return a;
}
