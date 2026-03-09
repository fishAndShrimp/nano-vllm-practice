# Frequent Errors

## __syncthreads() after sdata 

```cpp
if (lx < offset) {
    sdata[lx] += sdata[lx + offset];
}
__syncthreads(); 
```

---

## & in cudaMalloc

```cpp
float* d_A;
cudaMalloc((void**)&d_A, size);
```

---

## if(gx < size)

```cpp
if (gx < size) {
    sdata[lx] = a[gx];
} else {
    sdata[lx] = 0;
}
__syncthreads();
```

---

## if(lx < offset)

```cpp
int offset = blockDim.x / 2;
while (offset > 0) {
    if (lx < offset) {
        sdata[lx] += sdata[lx + offset];
    }
    __syncthreads();
    offset /= 2;
}
```

---

## ax bx mapping from lx

```cpp
for (int offset = 1; offset < blockDim.x; offset *= 2) {
    int ax = (2 * offset) * (lx + 1) - offset - 1;
    int bx = (2 * offset) * (lx + 1) - 1;
    if (bx < blockDim.x) {
        sdata[bx] += sdata[ax];
    }
    __syncthreads();
}
```

---

## independent m, independent sum, no if(lx == 0)

```cpp
scalar_t m = static_cast<scalar_t>(-INFINITY);
for (int phase = 0; blockDim.x * phase < size;
        phase++) {
    int gx = blockDim.x * phase + lx;
    sdata[lx] = (gx < size)
                    ? a[gx]
                    : static_cast<scalar_t>(-INFINITY);
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0;
            offset /= 2) {
        if (lx < offset) {
            sdata[lx] =
                max(sdata[lx], sdata[lx + offset]);
        }
        __syncthreads();
    }

    m = max(m, sdata[0]);
    __syncthreads();
}
```

---

## forget to update <<<1,1>>> placeholder

```cpp
GemmRowWiseKernel<scalar_t><<<1, 1>>>
```

---

## always use (ly, lx) in tile

```cpp
for (int k = 0; k < kTileSize; k++) {
    pvalue += a_tile[ly][k] * b_tile[k][lx];
}
```

```cpp
for (int k = 0; k < kTileSize; k++) {
    pvalues[lx] +=
        a_tile[k] * b_tile[k][lx];
}
```

---

## convert [gy][gx] => [gy*dim_m/dim_p + gx]

Always check the `gy*dim_m/dim_p` and `cond2` from if(cond1 && cond2)

```cpp
b_tile[row][col] =
    b[(kTileSize * phase + row) *
            dim_p +
        (kTileSize * tile_idx + col)];
```

---

