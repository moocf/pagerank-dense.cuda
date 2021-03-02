#pragma once
#include <array>

using namespace std;




// Defines an adjacency matrix (R:dst, C:src) based graph.
template <class T>
class DenseDiGraph {
  int *odeg;
  T   *ewt;
  int n;

  // Types
  public:
  using TKey    = int;
  using TVertex = void;
  using TEdge   = T;

  // Read operations
  public:
  int order() { return n; }
  int size()  { return n*n; }

  int* degrees() { return odeg; }
  T* edgeData()  { return ewt; }

  int degree(int u) { return odeg[u]; }
  T edgeData(int u, int v) { return ewt[n*v + u]; }
  void setEdgeData(int u, int v, T d) { ewt[n*v + u] = d; }

  // Write operations
  public:
  void addEdge(int u, int v, T d=1) {
    odeg[u] += d              != 0? 1 : 0;
    odeg[u] -= edgeData(u, v) != 0? 1 : 0;
    setEdgeData(u, v, d);
  }
  void removeEdge(int u, int v) { addEdge(u, v, 0); }

  // Lifetime operations
  public:
  DenseDiGraph(int _n) {
    n = _n;
    odeg = new int[n] {};
    ewt  = new T[n*n] {};
  }

  ~DenseDiGraph() {
    delete[] odeg;
    delete[] ewt;
  }
};
