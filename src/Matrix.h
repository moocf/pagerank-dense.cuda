#pragma once


template <typename T>
class Matrix {
  T *data;
  int rows;
  int cols;

  Matrix(int r, int c) {
    data = new T[r*c];
    rows = r;
    cols = c;
  }
};
