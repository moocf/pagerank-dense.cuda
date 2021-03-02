#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include "DenseDiGraph.h"

using std::string;
using std::ifstream;
using std::istringstream;
using std::getline;




auto readMtx(string pth) {
  int r, c, sz;
  ifstream f(pth);
  string ln;

  getline(f, ln);
  getline(f, ln);
  istringstream ls(ln);
  ls >> r >> c >> sz;
  DenseDiGraph<float> a(r);
  while (getline(f, ln)) {
    int i, j; float w;
    ls = istringstream(ln);
    if (!(ls >> i >> j >> w)) break;
    if (w > 0) a.addEdge(i-1, j-1, w);
  }
  return a;
}
