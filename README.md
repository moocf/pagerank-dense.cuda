PageRank (PR) algorithm for dense graphs.

<br>

```cpp
// DenseDiGraph:
DenseDiGraph(N);
g.order;
g.degrees[];
g.weights[];
g.weight(i, j);
g.setWeight(i, j, v);
g.setLink(i, j, w);

// degree:
degree(g, i);

// normalizeDegree:
normalizeDegree(g);



// ceilDiv:
ceilDiv(x, y);

// print:
print(x, N);
print(x, R, C);

// readMtx:
readMtx(pth);



// fill:
fill(x, N, v);
fill(x, v);
fillOmp(x, N, v);
fillOmp(x, v);
fillKernelLoop(x, N, v, i, DI);
fillKernel(x, N, v);
fillCuda(x, N, v);
fillCuda(x, v);

// sum:
sum(x, N);
sum(x);
sumOmp(x, N);
sumOmp(x);
sumKernelReduce(a, N, i);
sumKernelLoop(x, N, i, DI);
sumKernel(a, x, N);
sumCuda(x, N);
sumCuda(x);

// errorAbs:
errorAbs(x, y, N);
errorAbs(x, y);
errorAbsOmp(x, y, N);
errorAbsOmp(x, y);
errorAbsKernelLoop(x, y, N, i, DI);
errorAbsKernel(a, x, y, N);
errorAbsCuda(x, y, N);
errorAbsCuda(x, y);

// dotProduct:
dotProduct(x, y, N);
dotProduct(x, y);
dotProductOmp(x, y, N);
dotProductOmp(x, y);
dotProductKernelLoop(x, y, N, i, DI);
dotProductKernel(a, x, y, N);
dotProductCuda(x, y, N);
dotProductCuda(x, y);

// pageRank:
pageRank(a, w, N, p, E);
pageRank(a, w, N, o);
pageRank(a, g, o);
pageRankOmp(a, w, N, p, E);
pageRankOmp(a, w, N, o);
pageRankOmp(a, g, o);
pageRankKernel(e, a, r, w, N, p, E);
pageRankCuda(a, w, N, p, E);
pageRankCuda(a, w, N, o);
pageRankCuda(a, g, o);
```
