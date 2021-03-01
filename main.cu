#include <stdio.h>
#include "src/main.h"
#include "tests.h"
#include "runs.h"

using namespace std;




template <class T>
void runPageRank(DenseDiGraph<T>& g) {
  int N = g.order;
  float *ranks = new float[N], t;
  normalizeDegree(g);
  t = measureDuration([&]() { pageRank(ranks, g); });
  printf("[%07.1f ms] pageRank     = \n", t); // print(ranks, N);
  t = measureDuration([&]() { pageRankOmp(ranks, g); });
  printf("[%07.1f ms] pageRankOmp  = \n", t); // print(ranks, N);
  t = measureDuration([&]() { pageRankCuda(ranks, g); });
  printf("[%07.1f ms] pageRankCuda = \n", t); // print(ranks, N);
  delete[] ranks;
}


int main(int argc, char **argv) {
  printf("Loading graph ...\n");
  auto g = readMtx(argv[1]);
  testAll();
  runFill();
  runSum();
  runErrorAbs();
  runDotProduct();
  runPageRank(g);
  return 0;
}
