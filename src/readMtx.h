#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include "DenseDiGraph.h"

using namespace std;




DenseDiGraph<float> readMtx(string pth) {
  string ln;
  ifstream f(pth);

  // skip 1st line
  getline(f, ln);

  // read 2nd line
  int r, c, sz;
  getline(f, ln);
  istringstream ls(ln);
  ls >> r >> c >> sz;
  DenseDiGraph<float> a(r);

  // read remaining lines (edges)
  while (getline(f, ln)) {
    int i, j; float w;
    ls = istringstream(ln);
    if (!(ls >> i >> j >> w)) break;
    a.addLink(i-1, j-1, w);
  }
  return a;
}
