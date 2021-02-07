#pragma once
#include <array>

using namespace std;


// Defines an adjacency matrix (R:dst, C:src) based graph.
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

  void addLink(int i, int j, T w=1) {
    int N = order;
    T* w0 = &weights[N*j + i];
    degrees[i] += w   != 0? 1 : 0;
    degrees[i] -= *w0 != 0? 1 : 0;
    *w0 = w;
  }
};
