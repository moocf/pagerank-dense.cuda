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

  inline T weight(int i, int j) {
    int N = order;
    return weights[N*j + i];
  }

  inline void setWeight(int i, int j, T v) {
    int N = order;
    weights[N*j + i] = v;
  }

  inline void addLink(int i, int j, T w=1) {
    degrees[i] += w            != 0? 1 : 0;
    degrees[i] -= weight(i, j) != 0? 1 : 0;
    setWeight(i, j, w);
  }
};
