const libspropack = joinpath(Pkg.dir("PROPACK"), "deps", "PROPACK", "single", "libspropack")
const libdpropack = joinpath(Pkg.dir("PROPACK"), "deps", "PROPACK", "double", "libdpropack")
const libcpropack = joinpath(Pkg.dir("PROPACK"), "deps", "PROPACK", "complex8", "libcpropack")
const libzpropack = joinpath(Pkg.dir("PROPACK"), "deps", "PROPACK", "complex16", "libzpropack")

for (fname, lname, elty) in ((:slansvd_, :libspropack, Float32),
                             (:dlansvd_, :libdpropack, Float64))
    @eval begin
        """lansvd!: Compute leading singular triplets.

        - jobu: compute left singular vectors ('Y'/'N')
        - jobv: compute right singular vectors ('Y'/'N')
        - m: number of rows of A
        - n: number of cols of A
        - kmax: maximum number of iterations (= max dimension of Krylov space)
        - aprod: function defining the linear operator A
        - U: m x k array to store left singular vectors (k ≤ min(kmax,m,n))
        - s: length k array to store desired singular values (k ≤ min(kmax,m,n))
        - bnd: length k array to store error estimates on the singular values
        - V: n x k array to store right singular vectors
        - tolin: desired accuracy; the error on s[i] is approximately
            max(16ϵ s[1], tolin * s[i])
        - work: work array of size
            - m + n + 9*kmax + 2*kmax^2 + 4 + max(m+n, 4*kmax+4) if jobu = jobv = 'N'
            - m + n + 9*kmax + 5*kmax^2 + 4 + max(3*kmax^2 + 4*kmax+4, nb*max(m,n)) otherwise
              where nb is a BLAS-3 block size
        - iwork: integer work array of size
            - 2*kmax + 1 if jobu = jobv = 'N'
            - 8*kmax     otherwise
        - doption: [δ, η, ‖A‖], where
            - δ: the level of orthogonality desired,
            - η: vectors with components larger than η will be purged during reorthogonalization
            - ‖A‖: estimate of the norm of A
        - ioption: [cgs, elr], where
            - cgs: 1 = classical Gram-Schmidt, 0 = modified Gram-Schmidt
            - elr: 1 = extended local orthogonality
        - dparm: array for passing data to aprod
        - iparm: array for passing integer data to aprod.

        Returns: (U, s, V, bnd).
        """
        function lansvd!(jobu::Char, jobv::Char, m::Integer, n::Integer,
            kmax::Integer, aprod, U::Array{$elty,2}, s::Vector{$elty},
            bnd::Vector{$elty}, V::Array{$elty,2}, tolin::$elty,
            work::Vector{$elty}, iwork::Vector{Int32}, doption::Vector{$elty},
            ioption::Vector{Int32}, dparm::Vector{$elty}, iparm::Vector{Int32})

            # extract values
            # in both Fortran and Julia, arrays are column major
            k = length(s)
            ldu, ku = size(U)
            ldv, kv = size(V)

            # check
            k <= kmax || error("too many triplets requested")
            ldu >= m || error("U must have m = $m rows")
            ku >= kmax+1 || error("U must have kmax+1 columns (kmax=$kmax)")
            ldv >= n || error("V must have n = $n rows")
            kv >= kmax || error("V must have kmax columns (kmax=$kmax)")

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

            info[1] == 0 || error("lansvd return code: $(info[1])")

            return U, s, V, bnd
        end

        """lansvd: Compute leading singular triplets: simplified interface.

        - jobu: compute left singular vectors ('Y'/'N')
        - jobv: compute right singular vectors ('Y'/'N')
        - m: number of rows of A
        - n: number of cols of A
        - pff: pointer to function defining the linear operator A
        - initvec: starting vector for bidiagonalization
        - k: number of triplets desired
        - kmax: maximum number of iterations (= max dimension of Krylov space)
        - tolin: desired accuracy; the error on s[i] is approximately
            max(16ϵ s[1], tolin * s[i])
        """
        function lansvd(jobu::Char, jobv::Char, m::Integer, n::Integer, pff::Ptr{Void},
            initvec::Vector{$elty}, k::Integer, kmax::Integer, tolin::$elty)

            # Extract
            U = Array($elty, m, kmax + 1)
            copy!(U, 1, initvec, 1, m)
            s = Array($elty, k)
            bnd = Array($elty, k)
            V = Array($elty, n, kmax)

            nb = 16 # BLAS-3 blocking size. Don't know the size. It's almost surely a power of 2.
            if jobu == 'N' && jobv == 'N'
                lwork = Int32(m + n + 9kmax + 2kmax*kmax + 4 + max(m + n, 4kmax + 4))
                liwork = Int32(2kmax + 1)
            else
                lwork = Int32(m + n + 9kmax + 5kmax*kmax + 4 + max(3kmax*kmax + 4kmax + 4, nb*max(m, n)))
                liwork = Int32(8kmax)
            end
            work = Array($elty, lwork)
            iwork = Array(Int32, liwork)

            ϵ = eps($elty)
            doption = $elty[sqrt(ϵ/k); ϵ^(3/4)/sqrt(k); 0.0]  # propack will estimate ‖A‖
            ioption = Int32[0; 1]

            dparm = $elty[0]
            iparm = Int32[0]

            (U, s, V, bnd) = lansvd!(jobu, jobv, m, n, kmax, pff, U, s, bnd, V, tolin,
                work, iwork, doption, ioption, dparm, iparm)
            return (U[:,1:k], s, V[:,1:k], bnd)
        end
    end
end


for (fname, lname, elty) in ((:slansvd_irl_, :libspropack, Float32),
                             (:dlansvd_irl_, :libdpropack, Float64))
    @eval begin

        """lansvd_irl!: Compute leading singular triplets.

        - which: compute triplets for largest ('L') or smallest ('S') singular values
        - jobu: compute left singular vectors ('Y'/'N')
        - jobv: compute right singular vectors ('Y'/'N')
        - m: number of rows of A
        - n: number of cols of A
        - kmax: maximum number of iterations (= max dimension of Krylov space)
        - p: number of shifts per restart
        - maxiter: maximum number of restarts
        - aprod: function defining the linear operator A
        - U: m x k array to store left singular vectors (k ≤ min(kmax-p,m,n))
        - s: length k array to store desired singular values (k ≤ min(kmax-p,m,n))
        - bnd: length k array to store error estimates on the singular values
        - V: n x k array to store right singular vectors
        - tolin: desired accuracy; the error on s[i] is approximately
            max(16ϵ s[1], tolin * s[i])
        - work: work array of size
            - m + n + 9*kmax + 2*kmax^2 + 4 + max(m+n, 4*kmax+4) if jobu = jobv = 'N'
            - m + n + 9*kmax + 5*kmax^2 + 4 + max(3*kmax^2 + 4*kmax+4, nb*max(m,n)) otherwise
              where nb is a BLAS-3 block size
        - iwork: integer work array of size
            - 2*kmax + 1 if jobu = jobv = 'N'
            - 8*kmax     otherwise
        - doption: [δ, η, ‖A‖, gap], where
            - δ: the level of orthogonality desired,
            - η: vectors with components larger than η will be purged during reorthogonalization
            - ‖A‖: estimate of the norm of A
            - gap: smallest relative gap between the shifts and the lower bound on Ritz values
        - ioption: [cgs, elr], where
            - cgs: 1 = classical Gram-Schmidt, 0 = modified Gram-Schmidt
            - elr: 1 = extended local orthogonality
        - dparm: array for passing data to aprod
        - iparm: array for passing integer data to aprod.

        Returns: (U, s, V, bnd).
        """
        function lansvd_irl!(which::Char, jobu::Char, jobv::Char, m::Integer,
            n::Integer, kmax::Integer, p::Integer,
            maxiter::Integer, aprod, U::Array{$elty,2}, s::Vector{$elty},
            bnd::Vector{$elty}, V::Array{$elty,2}, tolin::$elty,
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
                 Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Void}, Ptr{$elty},
                 Ptr{Int32}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
                 Ptr{Int32}, Ptr{$elty}, Ptr{$elty}, Ptr{Int32},
                 Ptr{Int32}, Ptr{Int32}, Ptr{$elty}, Ptr{Int32},
                 Ptr{Int32}, Ptr{$elty}, Ptr{Int32}),
                &which, &jobu, &jobv, &m, &n,
                &kmax, &p, &k, &maxiter, aprod, U,
                &ldu, s, bnd, V,
                &ldv, &tolin, work, &length(work),
                iwork, &length(iwork), doption, ioption,
                info, dparm, iparm)

            info[1] == 0 || error("lansvd_irl return code: $(info[1])")

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
            V = Array($elty, n, kmax)

            nb = 16 # BLAS-3 blocking size. Don't know the size. It's almost surely a power of 2.
            if jobu == 'N' && jobv == 'N'
                lwork = Int32(m + n + 10kmax + 2kmax*kmax + 5 + max(m, max(n, 4kmax + 4)))
                liwork = Int32(2kmax + 1)
            else
                lwork = Int32(m + n + 10kmax + 5kmax*kmax + 4 + max(3kmax*kmax + 4kmax + 4, nb*max(m, n)))
                liwork = Int32(8kmax)
            end
            work = Array($elty, lwork)
            iwork = Array(Int32, liwork)

            ϵ = eps($elty)
            doption = $elty[sqrt(ϵ); ϵ^(3/4); 0.0; ϵ^(1/6)]
            ioption = Int32[0; 1]

            dparm = $elty[0]
            iparm = Int32[0]

            lansvd!(which, jobu, jobv, m, n, dim, p, neig, maxiter, pff, U, s,
                bnd, V, tolin, work, iwork, doption, ioption, dparm, iparm)
        end
    end
end
