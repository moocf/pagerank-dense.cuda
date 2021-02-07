#include <array>
#include <stdio.h>
#include "src/main.h"

using namespace std;


int main() {
  array<int, 4> x {1, 2, 3, 4};
  array<int, 4> y {1, 0, 1, 0};
  printf("dotProduct = %d\n", dotProduct(x, y));
  return 0;
}
