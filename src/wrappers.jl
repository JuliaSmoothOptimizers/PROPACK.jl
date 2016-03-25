const libspropack = joinpath(Pkg.dir("PROPACK"), "deps", "PROPACK", "single", "libspropack")
const libdpropack = joinpath(Pkg.dir("PROPACK"), "deps", "PROPACK", "double", "libdpropack")
const libcpropack = joinpath(Pkg.dir("PROPACK"), "deps", "PROPACK", "complex8", "libcpropack")
const libzpropack = joinpath(Pkg.dir("PROPACK"), "deps", "PROPACK", "complex16", "libzpropack")

for (fname, lname, elty) in ((:slansvd_, :libspropack, Float32),
                             (:dlansvd_, :libdpropack, Float64))
    @eval begin
        function lansvd!(jobu::Char, jobv::Char, m::Integer, n::Integer,
            kmax::Integer, aprod, U::StridedMatrix{$elty}, s::Vector{$elty},
            bnd::Vector{$elty}, V::StridedMatrix{$elty}, tolin::$elty,
            work::Vector{$elty}, iwork::Vector{Int32}, doption::Vector{$elty},
            ioption::Vector{Int32}, dparm::Vector{$elty}, iparm::Vector{Int32})

            # extract values
            k = length(s)
            ldu = stride(U, 2)
            ldv = stride(V, 2)

            # check

            # allocate
            info = Int32[0]

            ccall(($(string(fname)), $lname), Void,
                (Ptr{UInt8}, Ptr{UInt8}, Ptr{Int32}, Ptr{Int32},
                 Ptr{Int32}, Ptr{Int32}, Ptr{Void}, Ptr{$elty},
                 Ptr{Int32}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
                 Ptr{Int32}, Ptr{$elty}, Ptr{$elty}, Ptr{Int32},
                 Ptr{Int32}, Ptr{Int32}, Ptr{$elty}, Ptr{Int32},
                 Ptr{Int32}, Ptr{$elty}, Ptr{Int32}),
                &jobu, &jobv, &m, &n,
                &k, &kmax, aprod, U,
                &ldu, s, bnd, V,
                &ldv, &tolin, work, &length(work),
                iwork, &length(iwork), doption, ioption,
                info, dparm, iparm)

            if info[1] != 0
                error("info was $(info[1])")
            end

            return U, s, V, bnd
        end

        function lansvd(jobu::Char, jobv::Char, m::Integer, n::Integer, pff::Ptr{Void},
            initvec::Vector{$elty}, k::Integer, kmax::Integer, tolin::$elty)

            # Extract

            U = Array($elty, m, kmax + 1)
            copy!(U, 1, initvec, 1, m)
            s = Array($elty, k)
            bnd = Array($elty, k)
            V = Array($elty, m, kmax)

            nb = 10 # BLAS-3 blocking size. Don't know the size
            if jobu == 'N' && jobv == 'N'
                lwork = Int32(m + n + 9kmax + 2kmax*kmax + 4 + max(m + n, 4kmax + 4))
                liwork = Int32(2kmax + 1)
            else
                lwork = Int32(m + n + 9kmax + 5kmax*kmax + 4 + max(3kmax*kmax + 4kmax + 4, nb*max(m, n)))
                liwork = Int32(8kmax)
            end
            work = Array($elty, lwork)
            iwork = Array(Int32, liwork)

            doption = $elty[sqrt(eps($elty)); eps($elty)^(3/4); 0.0; zeros(7)]
            ioption = Int32[0; 1; zeros(8)]

            dparm = $elty[0]
            iparm = Int32[0]

            lansvd!(jobu, jobv, m, n, kmax, pff, U, s, bnd, V, tolin, work,
                iwork, doption, ioption, dparm, iparm)

        end
    end
end


for (fname, lname, elty) in ((:slansvd_irl_, :libspropack, Float32),
                             (:dlansvd_irl_, :libdpropack, Float64))
    @eval begin
        function lansvd_irl!(which::Char, jobu::Char, jobv::Char, m::Integer,
            n::Integer, dim::Integer, p::Integer, neig::Integer,
            maxiter::Integer, aprod, U::StridedMatrix{$elty}, s::Vector{$elty},
            bnd::Vector{$elty}, V::StridedMatrix{$elty}, tolin::$elty,
            work::Vector{$elty}, iwork::Vector{Int32}, doption::Vector{$elty},
            ioption::Vector{Int32}, dparm::Vector{$elty}, iparm::Vector{Int32})

            # extract values
            k = length(s)
            ldu = stride(U, 2)
            ldv = stride(V, 2)

            # check

            # allocate
            info = Int32[0]

            ccall(($(string(fname)), $lname), Void,
                (Ptr{UInt8}, Ptr{UInt8}, Ptr{UInt8}, Ptr{Int32}, Ptr{Int32},
                 Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Void}, Ptr{$elty},
                 Ptr{Int32}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
                 Ptr{Int32}, Ptr{$elty}, Ptr{$elty}, Ptr{Int32},
                 Ptr{Int32}, Ptr{Int32}, Ptr{$elty}, Ptr{Int32},
                 Ptr{Int32}, Ptr{$elty}, Ptr{Int32}),
                &which, &jobu, &jobv, &m, &n,
                &dim, &p, &neig, &maxiter, aprod, U,
                &ldu, s, bnd, V,
                &ldv, &tolin, work, &length(work),
                iwork, &length(iwork), doption, ioption,
                info, dparm, iparm)



            if info[1] != 0
                error("info was $(info[1])")
            end

            return U, s, V, bnd
        end

        function lansvd_irl!(which::Char, jobu::Char, jobv::Char, m::Integer,
            n::Integer, dim::Integer, p::Integer, neig::Integer,
            pff::Ptr{Void}, initvec::Vector{$elty}, k::Integer, kmax::Integer,
            tolin::$elty)

            # Extract

            U = Array($elty, m, kmax + 1)
            copy!(U, 1, initvec, 1, m)
            s = Array($elty, k)
            bnd = Array($elty, k)
            V = Array($elty, m, kmax)

            nb = 10 # BLAS-3 blocking size. Don't know the size
            if jobu == 'N' && jobv == 'N'
                lwork = Int32(m + n + 10kmax + 2kmax*kmax + 5 + max(m, max(n, 4kmax + 4)))
                liwork = Int32(2kmax + 1)
            else
                lwork = Int32(m + n + 10kmax + 5kmax*kmax + 4 + max(3kmax*kmax + 4kmax + 4, nb*max(m, n)))
                liwork = Int32(8kmax)
            end
            work = Array($elty, lwork)
            iwork = Array(Int32, liwork)

            doption = $elty[sqrt(eps($elty)); eps($elty)^(3/4); 0.0; zeros(7)]
            ioption = Int32[0; 1; zeros(8)]

            dparm = $elty[0]
            iparm = Int32[0]

            lansvd!(which, jobu, jobv, m, n, dim, p, neig, maxiter, pff, U, s,
                bnd, V, tolin, work, iwork, doption, ioption, dparm, iparm)
        end
    end
end
