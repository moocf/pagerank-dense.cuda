#pragma once
#include "_support.h"


class DenseDiGraph {
  public:
  int    order;
  int*   degrees;
  float* weights;

  DenseDiGraph(int n) {
    order = n;
    degrees = new int[n];
    weights = new float[n*n];
  }

  void addLink(int i, int j, float wt=1) {
    int n = order;
    degrees[i] += wt               != 0? 1 : 0;
    degrees[i] -= weights[j*n + i] != 0? 1 : 0;
    weights[j*n + i] = wt;
  }
};
