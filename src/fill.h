#pragma once
#include <array>
#include <vector>

using namespace std;


template <class T>
void fill(T *x, int N, T v) {
  for (int i=0; i<N; i++)
    x[i] = v;
}


template <class T, size_t N>
void fill(array<T, N>& x, T v) {
  fill(x.begin(), x.end(), v);
}


template <class T>
void fill(vector<T>& x, T v) {
  fill(x.begin(), x.end(), v);
}
