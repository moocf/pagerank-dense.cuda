#pragma once
#include "degree.h"




// Normalizes weights of graph by out-degree.
template <class G>
void normalizeDegree(G& x) {
  using E = typename G::TEdge;
  int N = x.order();
  for (int u=0; u<N; u++) {
    int d = degree(x, u);
    if (!d) d = N;
    for (int v=0; v<N; v++)
      if (x.edgeData(u, v) != 0 || d == N) x.setEdgeData(u, v, E(1)/d);
  }
}
