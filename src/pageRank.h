#pragma once
#include <utility>
#include <cmath>
#include <string.h>
#include "fill.h"
#include "DenseDiGraph.h"
#include "dotProduct.h"

using namespace std;


// Finds rank of nodes in graph.
template <class T>
void pageRank(T *a, DenseDiGraph<T>& x, float damping=0.85, float convergence=1e-5) {
  int N = x.order, e;
  fill(a, 1.0f/N);
  T *r0 = a, *r1 = new float[N];
  do {
    e = 0;
    for (int j=0; j<N; j++) {
      r1[j] = damping*dotProduct(x.weights+(N*j), r0, N) + (1-damping)/N;
      e += abs(r0[j] - r1[j]);
    }
    swap(r0, r1);
  } while (e > convergence);
  if (a != r0) memcpy(a, r0, N);
}
