#pragma once
#include "_support.h"
#include "DenseDiGraph.h"


// Finds out-degree of a node.
int degree(DenseDiGraph x, int i) {
  int a = 0;
  for (int j=0; j<x.order; j++)
    if (x.weights[j][i] !== 0) a++;
  return a;
}
