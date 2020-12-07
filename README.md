# PROPACK

| **Linux/macOS/Windows/FreeBSD** | **Coverage** |
|:-------------------------------:|:------------:|
| ![CI](https://github.com/JuliaSmoothOptimizers/PROPACK.jl/workflows/CI/badge.svg?branch=master)](https://github.com/JuliaSmoothOptimizers/PROPACK.jl/actions) [![Build Status](https://api.cirrus-ci.com/github/JuliaSmoothOptimizers/PROPACK.jl.svg)](https://cirrus-ci.com/github/JuliaSmoothOptimizers/PROPACK.jl) | [![codecov](https://codecov.io/gh/JuliaSmoothOptimizers/PROPACK.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/PROPACK.jl) |

A Julia interface to [PROPACK](http://sun.stanford.edu/~rmunk/PROPACK), a library for the computation of the truncated singular value decomposition of real (Float32, Float64) and complex (ComplexF32, ComplexF64) matrices or linear operators.
PROPACK only requires operator-vector products to estimate singular values and singular vectors.

## How to Install

```julia
julia> ]
pkg> add PROPACK
pkg> test PROPACK
```

## Examples

### Compute leading singular triplets

```julia
U, s, V, bnd, nprod, ntprod = tsvd(A, k=3)  # 3 largest singular values and their singular vectors
```

### Compute leading singular values only

```julia
s, bnd, nprod, ntprod = tsvdvals(A, k=3)
```

### Compute smallest singular triplets

Make sure `A` is square or short and wide to avoid the trailing zero singular values:

```julia
U, s, V, bnd, nprod, ntprod = tsvd_irl(A, k=2)
```

### Compute smallest singular values

```julia
s, bnd, nprod, ntprod = tsvdvals_irl(A, k=2)
```
