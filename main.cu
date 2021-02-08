#include <array>
#include <stdio.h>
#include "src/main.h"

using namespace std;




const char* testFill() {
  array<int, 4> a = {1, 2, 3, 4};

  fill(a, 4);
  for (auto& v : a)
    if (v != 4) return "fill";

  fillCuda(a, 4);
  for (auto& v : a)
    if (v != 4) return "fillCuda";
  return NULL;
}


const char* testSum() {
  array<int, 4> x = {1, 2, 3, 4};
  int a;

  a = sum(x);
  if (a != 10) return "sum";

  a = sumCuda(x);
  if (a != 10) return "sumCuda";
  return NULL;
}


const char* testDotProduct() {
  array<int, 4> x = {1, 2, 3, 4};
  array<int, 4> y = {1, 0, 1, 0};
  int a;

  a = dotProduct(x, y);
  if (a != 4) return "dotProduct";

  a = dotProductCuda(x, y);
  if (a != 4) return "dotProductCuda";
  return NULL;
}


const char* testErrorAbs() {
  array<int, 4> x = {1, 2, 3, 4};
  array<int, 4> y = {1, 1, 3, 5};
  int a;

  a = errorAbs(x, y);
  if (a != 2) return "errorAbs";

  a = errorAbsCuda(x, y);
  if (a != 2) return "errorAbsCuda";
  return NULL;
}


const char* testAll() {
  array<const char*, 4> ts = {
    testFill(),
    testSum(),
    testDotProduct(),
    testErrorAbs()
  };
  for (auto& t : ts) {
    if (!t) continue;
    printf("ERROR: %s() failed!\n", t);
    return t;
  }
  return NULL;
}




int main(int argc, char **argv) {
  testAll();
  auto g = readMtx(argv[1]);
  int N = g.order;
  float *ranks = new float[N];
  normalizeDegree(g);
  pageRank(ranks, g);
  printf("pageRank     = "); print(ranks, N);
  pageRankCuda(ranks, g);
  printf("pageRankCuda = "); print(ranks, N);
  return 0;
}
