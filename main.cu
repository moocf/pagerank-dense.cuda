#include <stdio.h>
#include "src/main.h"
#include "tests.h"
#include "runs.h"

using namespace std;




template <class T>
void runPageRank(DenseDiGraph<T>& g) {
  float t;
  auto ranks1 = pageRank(t, g);
  printf("[%07.1f ms] pageRank\n", t); print(ranks1);
  auto ranks2 = pageRankCuda(t, g);
  printf("[%07.1f ms] pageRankCuda\n", t); print(ranks2);
}


int main(int argc, char **argv) {
  testAll();
  printf("Loading graph ...\n");
  auto g = readMtx(argv[1]);
  normalizeDegree(g);
  print(g);
  // testAll();
  // runFill();
  // runSum();
  // runErrorAbs();
  // runDotProduct();
  runPageRank(g);
  return 0;
}
