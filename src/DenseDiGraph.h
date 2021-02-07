#pragma once
#include "_support.h"
#include <array>

using namespace std;


template <class T>
class DenseDiGraph {
  public:
  int order;
  int* degrees;
  T*   weights;

  DenseDiGraph(int n) {
    order = n;
    degrees = new int[n];
    weights = new T[n*n];
  }

  void addLink(int i, int j, float w=1) {
    int n = order;
    T& w0 = &weights[j*n + i];
    degrees[i] += w  != 0? 1 : 0;
    degrees[i] -= w0 != 0? 1 : 0;
    w0 = w;
  }
};
