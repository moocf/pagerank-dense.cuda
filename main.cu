#include <stdio.h>
#include "src/main.h"

using namespace std;


int main(int argc, char **argv) {
  auto g = readMtx(argv[1]);
  int N = g.order;
  float *ranks = new float[N];
  normalizeDegree(g);
  pageRank(ranks, g);
  printf("ranks = "); print(ranks, N);
  return 0;
}
