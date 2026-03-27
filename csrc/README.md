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

## forget to init in for loop

```cpp
for (int c; c < kDimHead; c++) {;}
for (int c=0; c < kDimHead; c++) {;}
```

---

## forget softmax

```python
out1 = q @ k.transpose(-2, -1)
out1 /= C**0.5
out1 = F.softmax(out1, dim=-1)
out1 = out1 @ v
```

---

## score 0.0 ruins sum_softmax

```cpp
        // [STEP: convert scores to weights]
#pragma unroll
        for (int lx = 0; lx < kTileSize; lx++) {
            // scores => weights
            // !!! [CRITICAL: MASKING] !!!
            // A rare bounds check required for math
            // correctness, not memory safety. Even without
            // array overflow, out-of-bounds default scores
            // (0.0) would evaluate to exp(0.0 - m_new) > 0,
            // silently corrupting the sum_softmax
            // denominator.

            auto gx = kTileSize * tile_idx + lx;
            if (gx < dim_t) {
                sw[lx] = exp(sw[lx] - m_new);
            } else {
                sw[lx] = 0.0;
            }

            sum_softmax += sw[lx];
        }
```

---

## iterate the full kDimHead when loading v with k_tile

```cpp
                // !!! [CRITICAL: REGISTER ALLOCATION] !!!
                // We must iterate the full kDimHead to
                // maintain static indexing. Dynamic
                // indexing would force `hidden` to spill to
                // slow HBM.
#pragma unroll
                for (int c = 0; c < kDimHead; c++) {
                    auto lc = c - kTileSize * phase;
                    if (0 <= lc && lc < kTileSize) {
                        hidden[c] +=
                            weight * sdata.v[lx][lc];
                    }
                }
```

---

## phase loop

```cpp
        // [STEP: phases]
        for (int phase = 0; kTileSize * phase < dim_c;
             phase++) {;}
```

---

## page_idx >= 0

```python
CORRECT:
i_block_table = [x for x in block_tables[i] if (x >= 0)]

ERROR:
i_block_table = [x for x in block_tables[i] if (x > 0)]
```

---

