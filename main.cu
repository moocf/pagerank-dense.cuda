#include <stdio.h>
#include "src/main.h"

using namespace std;


int main(int argc, char **argv) {
  auto g = readMtx(argv[1]);
  printf("order = %d\n", g.order);
  return 0;
}
