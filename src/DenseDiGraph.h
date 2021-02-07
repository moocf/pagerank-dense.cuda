#pragma once
#include "_support.h"
#include <vector>

using namespace std;


template <typename T>
class DenseDiGraph {
  public:
  int order;
  vector<int> degrees;
  vector<T>   weights;

  DenseDiGraph(int n) {
    order = n;
    degrees
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
