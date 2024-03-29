module PROPACK

export tsvd, tsvdvals, tsvd_irl, tsvdvals_irl

using LinearAlgebra
using LinearOperators
using PROPACK_jll

include("wrappers.jl")

# callback using dparm as passthrough pointer to save the linear operator
# thanks http://julialang.org/blog/2013/05/callback !
function __f__(
  transa_::Ptr{UInt8},
  m_::Ptr{Int},
  n_::Ptr{Int},
  x_::Ptr{T},
  y_::Ptr{T},
  dparm_::Ptr{T},
  iparm::Ptr{Int},
) where {T}
  m = unsafe_load(m_)
  n = unsafe_load(n_)
  dparm = reinterpret(Ptr{Nothing}, dparm_)
  transa = Char(unsafe_load(transa_))
  A = unsafe_pointer_to_objref(dparm)::AbstractLinearOperator
  (nargin, nargout) = transa == 'n' ? (n, m) : (m, n)
  x = unsafe_wrap(Array, x_, nargin)
  y = unsafe_wrap(Array, y_, nargout)
  transa == 'n' ? mul!(y, A, x) : mul!(y, A', x)
  nothing
end

"""
    tsvd(A::AbstractLinearOperator; kwargs...)

Compute a few leading singular triplets of `A` using only products with `A` and `A'`.

#### Keyword arguments
- `initvec::Vector`: initial vector for the iterations (default: zeros)
- `k::Integer`: number of leading singular values to approximate (default: 1)
- `kmax::Integer`: maximum dimensionality of search space (default: `min(size(A))`+10)
- `tolin::Real`: desired accuracy; the error on `s[i]` is approximately `max(16ϵ s[1], tolin * s[i])`
  (default: √ϵ).

#### Return values
- `U::Matrix`: orthogonal matrix of left singular vectors
- `s::Vector`: approximate leading singular values
- `V::Matrix`: orthogonal matrix of right singular vectors
- `bnd::Vector`: bound on the accuracy of each singular value
- `nprod::Int`: number of products with `A` required
- `ntprod::Int`: number of products with `A'` required.

The arrays `U`, `s` and `V` are such that `A - U * diagm(s) * V'` should be of the order of the next
largest singular value.
"""
function tsvd(
  A::AbstractLinearOperator{T};
  initvec::Vector{T} = zeros(T, size(A, 1)),
  k::Integer = 1,
  kmax::Integer = min(size(A)...) + 10,
  tolin::Real = sqrt(eps(real(one(T)))),
) where {T}
  __pf__ =
    @cfunction(__f__, Nothing, (Ptr{UInt8}, Ptr{Int}, Ptr{Int}, Ptr{T}, Ptr{T}, Ptr{T}, Ptr{Int}))

  m, n = size(A)
  dparm = pointer_from_objref(A)
  U, s, V, bnd = lansvd('Y', 'Y', m, n, __pf__, initvec, k, kmax, tolin, dparm)
  return (U, s, V, bnd, A.nprod, A.nctprod)
end

"""
    tsvdvals(A::AbstractLinearOperator; kwargs...)

Compute a few leading singular values of `A` using only products with `A` and `A'`.

#### Keyword arguments
See the documentation of `tsvd()`.

#### Return values
- `s::Vector`: approximate leading singular values
- `bnd::Vector`: bound on the accuracy of each singular value
- `nprod::Int`: number of products with `A` required
- `ntprod::Int`: number of products with `A'` required.
"""
function tsvdvals(
  A::AbstractLinearOperator{T};
  initvec::Vector{T} = zeros(T, size(A, 1)),
  k::Integer = 1,
  kmax::Integer = min(size(A)...) + 10,
  tolin::Real = sqrt(eps(real(one(T)))),
) where {T}
  __pf__ =
    @cfunction(__f__, Nothing, (Ptr{UInt8}, Ptr{Int}, Ptr{Int}, Ptr{T}, Ptr{T}, Ptr{T}, Ptr{Int}))

  m, n = size(A)
  dparm = pointer_from_objref(A)
  _, s, _, bnd = lansvd('N', 'N', m, n, __pf__, initvec, k, kmax, tolin, dparm)
  return (s, bnd, A.nprod, A.nctprod)
end

"""
    tsvd_irl(A::AbstractLinearOperator; kwargs...)

Compute a few extreme singular triplets of `A` using only products with `A` and `A'` and the
implicitly restarted Lanczos bidiagonalization method.

#### Keyword arguments
- `smallest::Bool`: whether the smallest or leading triplets should be approximated (default: `True`)
- `initvec::Vector`: initial vector for the iterations (default: zeros)
- `kmax::Integer`: maximum dimensionality of search space (default: `min(size(A))`+10)
- `p::Integer`: number of shifts per restart (default: 1)
- `k::Integer`: number of extreme singular values to approximate (default: 1)
- `maxiter::Integer`: maximum number of restarts (default: `min(size(A))`)
- `tolin::Real`: desired accuracy; the error on `s[i]` is approximately `max(16ϵ s[1], tolin * s[i])`
  (default: √ϵ).

#### Return values
- `U::Matrix`: orthogonal matrix of left singular vectors
- `s::Vector`: approximate leading singular values
- `V::Matrix`: orthogonal matrix of right singular vectors
- `bnd::Vector`: bound on the accuracy of each singular value
- `nprod::Int`: number of products with `A` required
- `ntprod::Int`: number of products with `A'` required.

If `smallest == False`, the arrays `U`, `s` and `V` are such that `A - U * diagm(s) * V'` should be
of the order of the next largest singular value.

If `smallest == True`, `U * diagm(s) * V'` is (an approximation of) the best rank-`k` approximation
of `A`.
"""
function tsvd_irl(
  A::AbstractLinearOperator{T};
  smallest::Bool = true,
  initvec::Vector{T} = zeros(T, size(A, 1)),
  kmax::Integer = min(size(A)...) + 10,
  p::Integer = 1,
  k::Integer = 1,
  maxiter::Integer = min(size(A)...),
  tolin::Real = sqrt(eps(real(one(T)))),
) where {T}
  __pf__ =
    @cfunction(__f__, Nothing, (Ptr{UInt8}, Ptr{Int}, Ptr{Int}, Ptr{T}, Ptr{T}, Ptr{T}, Ptr{Int}))

  m, n = size(A)
  dparm = pointer_from_objref(A)
  U, s, V, bnd = lansvd_irl(
    smallest ? 'S' : 'L',
    'Y',
    'Y',
    m,
    n,
    kmax,
    p,
    k,
    maxiter,
    __pf__,
    initvec,
    tolin,
    dparm,
  )
  return (U, s, V, bnd, A.nprod, A.nctprod)
end

"""
    tsvdvals_irl(A::AbstractLinearOperator; kwargs...)

Compute a few extreme singular values of `A` using only products with `A` and `A'` and the
implicitly restarted Lanczos bidiagonalization method.

#### Keyword arguments
See the documentation of `tsdv_irl()`.

#### Return values
- `s::Vector`: approximate leading singular values
- `bnd::Vector`: bound on the accuracy of each singular value
- `nprod::Int`: number of products with `A` required
- `ntprod::Int`: number of products with `A'` required.
"""
function tsvdvals_irl(
  A::AbstractLinearOperator{T};
  smallest::Bool = true,
  initvec::Vector{T} = zeros(T, size(A, 1)),
  kmax::Integer = min(size(A)...) + 10,
  p::Integer = 1,
  k::Integer = 1,
  maxiter::Integer = min(size(A)...),
  tolin::Real = sqrt(eps(real(one(T)))),
) where {T}
  __pf__ =
    @cfunction(__f__, Nothing, (Ptr{UInt8}, Ptr{Int}, Ptr{Int}, Ptr{T}, Ptr{T}, Ptr{T}, Ptr{Int}))

  m, n = size(A)
  dparm = pointer_from_objref(A)
  _, s, _, bnd = lansvd_irl(
    smallest ? 'S' : 'L',
    'N',
    'N',
    m,
    n,
    kmax,
    p,
    k,
    maxiter,
    __pf__,
    initvec,
    tolin,
    dparm,
  )
  return (s, bnd, A.nprod, A.nctprod)
end

# interface for matrices
for fname in (:tsvd, :tsvdvals, :tsvd_irl, :tsvdvals_irl)
  @eval $fname(A::AbstractMatrix{T}; kwargs...) where {T} = $fname(LinearOperator(A); kwargs...)
end

end # module
