#pragma once
#include <utility>
#include <cmath>
#include <string.h>
#include "_support.h"
#include "fill.h"
#include "DenseDiGraph.h"
#include "dotProduct.h"

using namespace std;


template <class T>
struct pageRankOptions {
  T damping;
  T convergence;

  pageRankOptions(T _damping=T(0.85), T _convergence=T(1e-6)) {
    damping = _damping;
    convergence = _convergence;
  }
};




// Finds rank of nodes in graph.
template <class T>
void pageRank(T *a, T *w, int N, T p=0.85, T E=1e-6) {
  fill(a, N, 1.0/N);
  T *r0 = a;
  T *r1 = new T[N];
  while (1) {
    int es = 0;
    for (int j=0; j<N; j++) {
      r1[j] = p*dotProduct(&w[N*j], r0, N) + (1-p)/N;
      T e = abs(r0[j] - r1[j]);
      if (e >= E) es++;
    }
    swap(r0, r1);
    if (!es) break;
  }
  if (a != r0) memcpy(a, r0, N*sizeof(T));
}


template <class T>
void pageRank(T *a, T *w, int N, pageRankOptions<T> o) {
  pageRank(a, w, N, o.damping, o.convergence);
}


template <class T>
void pageRank(T *a, DenseDiGraph<T>& x, pageRankOptions<T> o) {
  pageRank(a, x.weights, x.order, o);
}




template <class T>
__global__ void pageRankKernel(int *es, T *r0, T *r1, T *w, int N, T p) {
  DEFINE(tx, ty, bx, by, BX, BY, GX, GY);

  for (int i=bx; i<N; i+=GX) {
    T wir = dotProductKernel(&w[N*i], r0, N);
    r1[i] = p*wir + (1-p)/N;
    T e = abs(r1[i] - r0[i]);
    if (e >= E) atomicAdd(&es, 1);
  }
}
