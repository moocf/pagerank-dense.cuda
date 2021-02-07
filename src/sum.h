#pragma once
#include <array>
#include <vector>


template <class T>
T sum(T *x, int N) {
  T a = T();
  for (int i=0; i<N; i++)
    a += x[i];
  return a;
}


template <class T, size_t N>
T sum(array<T, N>& x) {
  T a = T();
  for (auto& v : x)
    a += v;
  return a;
}


template <class T>
T sum(vector<T>& x) {
  T a = T();
  for (auto& v : x)
    a += v;
  return a;
}
