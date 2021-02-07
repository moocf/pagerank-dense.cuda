#include <stdio.h>
#include "src/main.h"

using namespace std;


int main() {
  array<int, 4> a {1, 2, 3, 4};
  printf("sum = %d\n", sum(a));
  return 0;
}
