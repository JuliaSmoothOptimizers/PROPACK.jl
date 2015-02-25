const libdpropack = joinpath(Pkg.dir("PROPACK"), "deps", "PROPACK", "double", "libdpropack")

function lansvd!(jobu::Char, jobv::Char, m::Integer, n::Integer,
    kmax::Integer, aprod, U::StridedMatrix{Float64}, s::Vector{Float64},
    bnd::Vector{Float64}, V::StridedMatrix{Float64}, tolin::Float64,
    work::Vector{Float64}, iwork::Vector{Int32}, doption::Vector{Float64},
    ioption::Vector{Int32}, dparm::Vector{Float64}, iparm::Vector{Int32})

    # extract values
    k = length(s)
    ldu = stride(U, 2)
    ldv = stride(V, 2)

    # check

    # allocate
    info = Int32[0]

    ccall((:dlansvd_, libdpropack), Void,
        (Ptr{UInt8}, Ptr{UInt8}, Ptr{Int32}, Ptr{Int32},
         Ptr{Int32}, Ptr{Int32}, Ptr{Void}, Ptr{Float64},
         Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
         Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32},
         Ptr{Int32}, Ptr{Int32}, Ptr{Float64}, Ptr{Int32},
         Ptr{Int32}, Ptr{Float64}, Ptr{Int32}),
        &jobu, &jobv, &m, &n,
        &k, &kmax, aprod, U,
        &ldu, s, bnd, V,
        &ldv, &tolin, work, &length(work),
        iwork, &length(iwork), doption, ioption,
        info, dparm, iparm)

    if info[1] != 0
        error("info was $(info[1])")
    end

    return U, s, V, bnd, info[1]
end

function lansvd(jobu::Char, jobv::Char, m::Integer, n::Integer, pff::Ptr{Void},
    initvec::Vector{Float64}, k::Integer, kmax::Integer, tolin::Float64)

    # Extract

    # pff = cfunction(ff, Void, (Ptr{UInt8}, Ptr{Int32}, Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32}))

    # if jobu == 'Y'
        U = Array(Float64, m, kmax + 1)
        copy!(U, 1, initvec, 1, m)
    # elseif jobu == 'N'
        # U = reshape(initvec, m, 1)
    # else
        # throw(ArgumentError("jobu must be either 'Y' or 'N'"))
    # end

    s = Array(Float64, k)

    bnd = Array(Float64, k)

    # if jobv == 'Y'
        V = Array(Float64, m, kmax)
    # elseif jobu == 'N'
        # V = Array(Float64, n, 0)
    # else
        # throw(ArgumentError("jobv must be either 'Y' or 'N'"))
    # end

    nb = 10 # BLAS-3 blocking size. Don't know the size
    # if jobu == 'N' && jobv == 'N'
        # lwork = int32(m + n + 9kmax + 2kmax*kmax + 4 + max(m + n, 4kmax + 4))
        # liwork = int32(2kmax + 1)
    # else
        lwork = int32(m + n + 9kmax + 5kmax*kmax + 4 + max(3kmax*kmax + 4kmax + 4, nb*max(m, n)))
        liwork = int32(8kmax)
    # end
    work = Array(Float64, lwork)
    iwork = Array(Int32, liwork)

    doption = Float64[sqrt(eps(Float64)); eps(Float64)^(3/4); 0.0; zeros(7)]
    ioption = Int32[0; 1; zeros(8)]

    dparm = Float64[0]
    iparm = Int32[0]

    lansvd!(jobu, jobv, m, n, kmax, pff, U, s, bnd, V, tolin, work,
        iwork, doption, ioption, dparm, iparm)

end
