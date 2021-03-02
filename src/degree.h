#pragma once




// Finds out-degree of a node.
template <class G>
int degree(G& x, int u) {
  int N = x.order(), a = 0;
  for (int v=0; v<N; v++)
    if (x.edgeData(u, v) != 0) a++;
  return a;
}
