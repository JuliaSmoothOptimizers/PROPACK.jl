# PROPACK

| **Travis, AppVeyor and Cirrus build statuses** | **Coverage** |
|:----------------------------------------------:|:------------:|
| [![Build Status](https://travis-ci.org/JuliaSmoothOptimizers/PROPACK.jl.svg?branch=master)](https://travis-ci.org/JuliaSmoothOptimizers/PROPACK.jl) [![Build status](https://ci.appveyor.com/api/projects/status/s065u6mwkbyuldmw?svg=true)](https://ci.appveyor.com/project/dpo/propack-jl) [![Build Status](https://api.cirrus-ci.com/github/JuliaSmoothOptimizers/Krylov.jl.svg)](https://cirrus-ci.com/github/JuliaSmoothOptimizers/Krylov.jl) | [![Coverage Status](https://coveralls.io/repos/github/JuliaSmoothOptimizers/PROPACK.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaSmoothOptimizers/PROPACK.jl?branch=master) |

A Julia interface to [PROPACK](http://sun.stanford.edu/~rmunk/PROPACK), a library for the computation of the truncated singular value decomposition of matrices or linear operators.
PROPACK only requires operator-vector products to estimate singular values and singular vectors.

## How to Install

```julia
julia> ]
pkg> add PROPACK
pkg> build PROPACK
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
