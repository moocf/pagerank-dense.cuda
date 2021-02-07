#pragma once
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "arrayFill.h"
#include "DenseDiGraph.h"
#include "dotProduct.h"


// Finds rank of nodes in graph.
void pageRank(float *a, DenseDiGraph x, float damping=0.85, float convergence=1e-5) {
  int n = x.order;
  float *r0 = arrayFill(a, n, 1.0f/n);
  float *r1 = new float[n];
  do {
    int e = 0;
    for (int j=0; j<n; j++) {
      r1[j] = damping*dotProduct(x.weights+(j*n), r0, n) + (1-damping)/n;
      e += fabs(r0[j] - r1[j]);
    }
    memcpy(r0, r1, n);
  } while (e > convergence);
  return r0;
}
