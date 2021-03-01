#pragma once
#include <vector>
#include <stdio.h>
#include "src/main.h"

using namespace std;




const char* testFill() {
  vector<int> x {1, 2, 3, 4};
  vector<int> a(4);

  a = x;
  fill(a, 4);
  for (auto& v : a)
    if (v != 4) return "fill";

  a = x;
  fillOmp(a, 4);
  for (auto& v : a)
    if (v != 4) return "fillOmp";

  a = x;
  fillCuda(a, 4);
  for (auto& v : a)
    if (v != 4) return "fillCuda";
  return NULL;
}


const char* testSum() {
  vector<int> x {1, 2, 3, 4};
  int a;

  a = sum(x);
  if (a != 10) return "sum";

  a = sumOmp(x);
  if (a != 10) return "sumOmp";

  a = sumCuda(x);
  if (a != 10) return "sumCuda";
  return NULL;
}


const char* testDotProduct() {
  vector<int> x {1, 2, 3, 4};
  vector<int> y {1, 0, 1, 0};
  int a;

  a = dotProduct(x, y);
  if (a != 4) return "dotProduct";

  a = dotProductCuda(x, y);
  if (a != 4) return "dotProductCuda";
  return NULL;
}


const char* testErrorAbs() {
  vector<int> x {1, 2, 3, 4};
  vector<int> y {1, 1, 3, 5};
  int a;

  a = errorAbs(x, y);
  if (a != 2) return "errorAbs";

  a = errorAbsCuda(x, y);
  if (a != 2) return "errorAbsCuda";
  return NULL;
}


void testAll() {
  vector<const char*> ts = {
    testFill(),
    testSum(),
    testDotProduct(),
    testErrorAbs()
  };
  int n = 0;
  for (auto& t : ts) {
    if (!t) continue;
    printf("ERROR: %s() failed!\n", t);
    n++;
  }
  printf("%d/%ld tests failed.\n", n, ts.size());
}
