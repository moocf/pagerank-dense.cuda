#pragma once
#include "_support.h"


class DenseDiGraph {
  int  order;
  int* degrees;
  int* weights;

  DenseDiGraph(int n) {
    order = n;
    degrees = new int[n];
    weights = new int[n*n];
  }

  void addLink(int i, int j, int wt=1) {
    int o = GET2D(weights, j, i, order);
    degrees[i] += wt         != 0? 1 : 0;
    degrees[i] -= weights[o] != 0? 1 : 0;
    weights[o] = wt;
  }
};
