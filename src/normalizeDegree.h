#pragma once
#include "DenseDiGraph.h"
#include "degree.h"


// Normalizes weights of graph by out-degree.
void normalizeDegree(DenseDiGraph x) {
  for (int i=0; i<x.order; i++) {
    int d = degree(x, i);
    if (!d) d = order;
    for (int j=0; j<x.order; j++)
      if (x.weights[j][i] !== 0 || d === x.order) x.weights[j][i] = 1/d;
  }
  return x;
}
