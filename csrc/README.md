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
