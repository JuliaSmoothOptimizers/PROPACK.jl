module PROPACK

export tsvd, tsvdvals, tsvd_irl, tsvdvals_irl

include("wrappers.jl")

# callback using dparm as passthrough pointer to save the linear operator
# thanks http://julialang.org/blog/2013/05/callback !
function __f__{T}(transa_::Ptr{UInt8}, m_::Ptr{Int32}, n_::Ptr{Int32},
                  x_::Ptr{T}, y_::Ptr{T}, dparm_::Ptr{T}, iparm::Ptr{Int32})
  m = unsafe_load(m_)
  n = unsafe_load(n_)
  dparm = reinterpret(Ptr{Void}, dparm_)
  transa = Char(unsafe_load(transa_))
  A = unsafe_pointer_to_objref(dparm)::AbstractMatrix
  (nargin, nargout) = transa == 'n' ? (n, m) : (m, n)
  x = VERSION < v"0.5" ? pointer_to_array(x_, nargin) : unsafe_wrap(Array, x_, nargin)
  y = transa == 'n' ? (A * x) : (A' * x)
  unsafe_copy!(y_, pointer(y), nargout)
  nothing
end

function tsvd{T}(A::AbstractMatrix{T};
                 initvec::Vector{T} = zeros(T, size(A, 1)), k::Integer = 1,
                 kmax::Integer = min(size(A)...)+10,
                 tolin::Real = sqrt(eps(real(one(T)))))

    __pf__ = cfunction(__f__, Void, (Ptr{UInt8}, Ptr{Int32}, Ptr{Int32}, Ptr{T}, Ptr{T}, Ptr{T}, Ptr{Int32}))

    m, n = size(A)
    dparm = pointer_from_objref(A)
    lansvd('Y', 'Y', m, n, __pf__, initvec, k, kmax, tolin, dparm)
end

function tsvdvals{T}(A::AbstractMatrix{T};
                     initvec::Vector{T} = zeros(T, size(A, 1)), k::Integer = 1,
                     kmax::Integer = min(size(A)...)+10,
                     tolin::Real = sqrt(eps(real(one(T)))))

    __pf__ = cfunction(__f__, Void, (Ptr{UInt8}, Ptr{Int32}, Ptr{Int32}, Ptr{T}, Ptr{T}, Ptr{T}, Ptr{Int32}))

    m, n = size(A)
    dparm = pointer_from_objref(A)
    _, s, _, bnd = lansvd('N', 'N', m, n, __pf__, initvec, k, kmax, tolin, dparm)
    return s, bnd
end

function tsvd_irl{T}(A::AbstractMatrix{T};
                     smallest::Bool = true, initvec::Vector{T} = zeros(T, size(A, 1)),
                     kmax::Integer = min(size(A)...)+10,
                     p::Integer = 1, k::Integer = 1,
                     maxiter::Integer = min(size(A)...), tolin::Real = sqrt(eps(real(one(T)))))

    __pf__ = cfunction(__f__, Void, (Ptr{UInt8}, Ptr{Int32}, Ptr{Int32}, Ptr{T}, Ptr{T}, Ptr{T}, Ptr{Int32}))

    m, n = size(A)
    dparm = pointer_from_objref(A)
    lansvd_irl(smallest ? 'S' : 'L', 'Y', 'Y', m, n, kmax, p, k, maxiter, __pf__, initvec, tolin, dparm)
end

function tsvdvals_irl{T}(A::AbstractMatrix{T};
                         smallest::Bool = true, initvec::Vector{T} = zeros(T, size(A, 1)),
                         kmax::Integer = min(size(A)...)+10,
                         p::Integer = 1, k::Integer = 1,
                         maxiter::Integer = min(size(A)...), tolin::Real = sqrt(eps(real(one(T)))))
    __pf__ = cfunction(__f__, Void, (Ptr{UInt8}, Ptr{Int32}, Ptr{Int32}, Ptr{T}, Ptr{T}, Ptr{T}, Ptr{Int32}))

    m, n = size(A)
    dparm = pointer_from_objref(A)
    _, s, _, bnd = lansvd_irl(smallest ? 'S' : 'L', 'N', 'N', m, n, kmax, p, k, maxiter, __pf__, initvec, tolin, dparm)
    return s, bnd
end

end # module
