# PROPACK

| **Documentation** | **Linux/macOS/Windows/FreeBSD** | **Coverage** | **DOI** |
|:-----------------:|:-------------------------------:|:------------:|:-------:|
| [![docs-stable][docs-stable-img]][docs-stable-url] [![docs-dev][docs-dev-img]][docs-dev-url] | [![build-gh][build-gh-img]][build-gh-url] [![build-cirrus][build-cirrus-img]][build-cirrus-url] | [![codecov][codecov-img]][codecov-url] | [![doi][doi-img]][doi-url] |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://JuliaSmoothOptimizers.github.io/PROPACK.jl/stable
[docs-dev-img]: https://img.shields.io/badge/docs-dev-purple.svg
[docs-dev-url]: https://JuliaSmoothOptimizers.github.io/PROPACK.jl/dev
[build-gh-img]: https://github.com/JuliaSmoothOptimizers/PROPACK.jl/workflows/CI/badge.svg?branch=main
[build-gh-url]: https://github.com/JuliaSmoothOptimizers/PROPACK.jl/actions
[build-cirrus-img]: https://img.shields.io/cirrus/github/JuliaSmoothOptimizers/PROPACK.jl?logo=Cirrus%20CI
[build-cirrus-url]: https://cirrus-ci.com/github/JuliaSmoothOptimizers/PROPACK.jl
[codecov-img]: https://codecov.io/gh/JuliaSmoothOptimizers/PROPACK.jl/branch/main/graph/badge.svg
[codecov-url]: https://app.codecov.io/gh/JuliaSmoothOptimizers/PROPACK.jl
[doi-img]: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.3572724-blue.svg
[doi-url]: https://doi.org/10.5281/zenodo.3572724

A Julia interface to [PROPACK](http://sun.stanford.edu/~rmunk/PROPACK), a library for the computation of the truncated singular value decomposition of real (Float32, Float64) and complex (ComplexF32, ComplexF64) matrices or linear operators.
PROPACK only requires operator-vector products to estimate singular values and singular vectors.

## Reference

> Larsen Rasmus M. (1998).
> Lanczos bidiagonalization with partial reorthogonalization.
> Department of Computer Science, Aarhus University, Technical report, DAIMI PB-357.

## How to Cite

If you use PROPACK.jl in your work, please cite using the format given in [`CITATION.bib`](https://github.com/JuliaSmoothOptimizers/PROPACK.jl/blob/main/CITATION.bib).

## How to Install

```julia
julia> ]
pkg> add PROPACK
pkg> test PROPACK
```

The version 0.5.0 of PROPACK.jl requires at least Julia 1.8.

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

## Bug reports and discussions

If you think you found a bug, feel free to open an [issue](https://github.com/JuliaSmoothOptimizers/PROPACK.jl/issues).
Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.

If you want to ask a question not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers) organization, so questions about any of our packages are welcome.
