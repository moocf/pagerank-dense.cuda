#pragma once
#include "DenseDiGraph.h"
#include "dotProduct.h"


// Finds rank of nodes in graph.
void pageRank(float *a, DenseDiGraph x, float damping=0.85, float convergence=1e-5) {
  int n = x.order;
  float *r = new float[n];
  var ranks = new Array(order).fill(0).map(() => 1/order);
  do {
    var r = ranks.slice(), e = 0;
    for (var j=0; j<order; j++) {
      r[j] = damping*dotProduct(weights[j], ranks) + (1-damping)/order;
      e += Math.abs(ranks[j] - r[j]);
    }
    ranks = r;
  } while (e > convergence);
  return ranks;
}
module.exports = pageRank;
