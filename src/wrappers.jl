"""
lansvd!: Compute leading singular triplets.

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
- cwork: complex work array
- doption: [δ, η, ‖A‖], where
    - δ: the level of orthogonality desired,
    - η: vectors with components larger than η will be purged during reorthogonalization
    - ‖A‖: estimate of the norm of A
- ioption: [cgs, elr], where
    - cgs: 1 = classical Gram-Schmidt, 0 = modified Gram-Schmidt
    - elr: 1 = extended local orthogonality
- dparm: array for passing data to aprod; WARNING: we use dparm as passthrough pointer!!!
- iparm: array for passing integer data to aprod.

Returns: (U, s, V, bnd).
"""
function lansvd! end

"""
lansvd: Compute leading singular triplets: simplified interface.

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
function lansvd end

"""
lansvd_irl!: Compute leading singular triplets.

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
- cwork: complex work array
- doption: [δ, η, ‖A‖, gap], where
    - δ: the level of orthogonality desired,
    - η: vectors with components larger than η will be purged during reorthogonalization
    - ‖A‖: estimate of the norm of A
    - gap: smallest relative gap between the shifts and the lower bound on Ritz values
- ioption: [cgs, elr], where
    - cgs: 1 = classical Gram-Schmidt, 0 = modified Gram-Schmidt
    - elr: 1 = extended local orthogonality
- dparm: array for passing data to aprod; WARNING: we use dparm as passthrough pointer!!!
- iparm: array for passing integer data to aprod.

Returns: (U, s, V, bnd).
"""
function lansvd_irl! end

"""
lansvd_irl: Compute leading singular triplets, simplified interface.

- which: compute triplets for largest ('L') or smallest ('S') singular values
- jobu: compute left singular vectors ('Y'/'N')
- jobv: compute right singular vectors ('Y'/'N')
- m: number of rows of A
- n: number of cols of A
- kmax: maximum number of iterations (= max dimension of Krylov space)
- p: number of shifts per restart
- k: number of triplets desired
- maxiter: maximum number of restarts
- pff: pointer to function defining the linear operator A
- initvec: starting vector for bidiagonalization
- tolin: desired accuracy; the error on s[i] is approximately
    max(16ϵ s[1], tolin * s[i]).

Returns: (U, s, V, bnd).
"""
function lansvd_irl end

for (fname, fname2, lname, elty, subty) in (
  (:slansvd_, :slansvd_irl_, :libspropack, :Float32, :Float32),
  (:dlansvd_, :dlansvd_irl_, :libdpropack, :Float64, :Float64),
  (:clansvd_, :clansvd_irl_, :libcpropack, :ComplexF32, :Float32),
  (:zlansvd_, :zlansvd_irl_, :libzpropack, :ComplexF64, :Float64),
)
  @eval begin
    function lansvd!(
      jobu::Char,
      jobv::Char,
      m::Integer,
      n::Integer,
      kmax::Integer,
      aprod,
      U::Matrix{$elty},
      s::Vector{$subty},
      bnd::Vector{$subty},
      V::Matrix{$elty},
      tolin::$subty,
      work::Vector{$subty},
      iwork::Vector{Int},
      doption::Vector{$subty},
      ioption::Vector{Int},
      dparm::Ptr{Nothing},
      iparm::Vector{Int};
      cwork::Vector{$elty} = $elty[],
    )

      # extract values
      # in both Fortran and Julia, arrays are column major
      k = length(s)
      ldu, ku = size(U)
      ldv, kv = size(V)
      lwork = length(work)
      liwork = length(iwork)
      lcwork = length(cwork)

      # check
      k <= kmax || error("too many triplets requested")
      ldu >= m || error("U must have m = $m rows")
      ku >= kmax + 1 || error("U must have kmax+1 columns (kmax=$kmax)")
      ldv >= n || error("V must have n = $n rows")
      kv >= kmax || error("V must have kmax columns (kmax=$kmax)")

      # allocate
      info = Int[0]

      if $elty <: Real
        @ccall $lname.$fname(
          jobu::Ref{UInt8},
          jobv::Ref{UInt8},
          m::Ref{Int},
          n::Ref{Int},
          k::Ref{Int},
          kmax::Ref{Int},
          aprod::Ptr{Nothing},
          U::Ptr{$elty},
          ldu::Ref{Int},
          s::Ptr{$elty},
          bnd::Ptr{$subty},
          V::Ptr{$elty},
          ldv::Ref{Int},
          tolin::Ref{$subty},
          work::Ptr{$subty},
          lwork::Ref{Int},
          iwork::Ptr{Int},
          liwork::Ref{Int},
          doption::Ptr{$subty},
          ioption::Ptr{Int},
          info::Ptr{Int},
          dparm::Ptr{$elty},
          iparm::Ptr{Int},
        )::Cvoid
      elseif $elty <: Complex
        @ccall $lname.$fname(
          jobu::Ref{UInt8},
          jobv::Ref{UInt8},
          m::Ref{Int},
          n::Ref{Int},
          k::Ref{Int},
          kmax::Ref{Int},
          aprod::Ptr{Nothing},
          U::Ptr{$elty},
          ldu::Ref{Int},
          s::Ptr{$elty},
          bnd::Ptr{$subty},
          V::Ptr{$elty},
          ldv::Ref{Int},
          tolin::Ref{$subty},
          work::Ptr{$subty},
          lwork::Ref{Int},
          cwork::Ptr{$elty},
          lcwork::Ref{Int},
          iwork::Ptr{Int},
          liwork::Ref{Int},
          doption::Ptr{$subty},
          ioption::Ptr{Int},
          info::Ptr{Int},
          dparm::Ptr{$elty},
          iparm::Ptr{Int},
        )::Cvoid
      end

      info[1] == 0 || error("lansvd return code: $(info[1])")

      return U, s, V, bnd
    end

    function lansvd(
      jobu::Char,
      jobv::Char,
      m::Integer,
      n::Integer,
      pff::Ptr{Nothing},
      initvec::Vector{$elty},
      k::Integer,
      kmax::Integer,
      tolin::$subty,
      dparm::Ptr{Nothing},
    )

      # Extract
      U = Matrix{$elty}(undef, m, kmax + 1)
      copyto!(U, 1, initvec, 1, m)
      s = Vector{$subty}(undef, k)
      bnd = Vector{$subty}(undef, k)
      V = Matrix{$elty}(undef, n, kmax)

      nb = 16 # BLAS-3 blocking size. Don't know the size. It's almost surely a power of 2.
      if jobu == 'N' && jobv == 'N'
        lwork = Int(m + n + 9kmax + 2kmax * kmax + 4 + max(m + n, 4kmax + 4))
        liwork = Int(2kmax + 1)
      else
        lwork =
          Int(m + n + 9kmax + 5kmax * kmax + 4 + max(3kmax * kmax + 4kmax + 4, nb * max(m, n)))
        liwork = Int(8kmax)
      end
      work = Vector{$subty}(undef, lwork)
      iwork = Vector{Int}(undef, liwork)
      if $elty <: Complex
        lcwork = Int(m + n + 32m)
        cwork = Vector{$elty}(undef, lcwork)
      else
        cwork = $elty[]
      end
      ϵ = eps($subty)
      doption = $subty[sqrt(ϵ / k); ϵ^(3 / 4) / sqrt(k); 0.0]  # propack will estimate ‖A‖
      ioption = Int[0; 1]

      iparm = Int[0]

      (U, s, V, bnd) = lansvd!(
        jobu,
        jobv,
        m,
        n,
        kmax,
        pff,
        U,
        s,
        bnd,
        V,
        tolin,
        work,
        iwork,
        doption,
        ioption,
        dparm,
        iparm,
        cwork = cwork,
      )
      return (U[:, 1:k], s, V[:, 1:k], bnd)
    end

    function lansvd_irl!(
      which::Char,
      jobu::Char,
      jobv::Char,
      m::Integer,
      n::Integer,
      kmax::Integer,
      p::Integer,
      maxiter::Integer,
      aprod,
      U::Matrix{$elty},
      s::Vector{$subty},
      bnd::Vector{$subty},
      V::Matrix{$elty},
      tolin::$subty,
      work::Vector{$subty},
      iwork::Vector{Int},
      doption::Vector{$subty},
      ioption::Vector{Int},
      dparm::Ptr{Nothing},
      iparm::Vector{Int};
      cwork::Vector{$elty} = $elty[],
    )

      # extract values
      # in both Fortran and Julia, arrays are column major
      k = length(s)
      ldu, ku = size(U)
      ldv, kv = size(V)
      lwork = length(work)
      liwork = length(iwork)
      lcwork = length(cwork)

      # check
      k <= kmax || error("too many triplets requested")
      ldu >= m || error("U must have m = $m rows")
      ku >= kmax + 1 || error("U must have kmax+1 columns (kmax=$kmax)")
      ldv >= n || error("V must have n = $n rows")
      kv >= kmax || error("V must have kmax columns (kmax=$kmax)")

      # allocate
      info = Int[0]

      if $elty <: Real
        @ccall $lname.$fname2(
          which::Ref{UInt8},
          jobu::Ref{UInt8},
          jobv::Ref{UInt8},
          m::Ref{Int},
          n::Ref{Int},
          kmax::Ref{Int},
          p::Ref{Int},
          k::Ref{Int},
          maxiter::Ref{Int},
          aprod::Ptr{Nothing},
          U::Ptr{$elty},
          ldu::Ref{Int},
          s::Ptr{$elty},
          bnd::Ptr{$subty},
          V::Ptr{$elty},
          ldv::Ref{Int},
          tolin::Ref{$subty},
          work::Ptr{$subty},
          lwork::Ref{Int},
          iwork::Ptr{Int},
          liwork::Ref{Int},
          doption::Ptr{$subty},
          ioption::Ptr{Int},
          info::Ptr{Int},
          dparm::Ptr{$elty},
          iparm::Ptr{Int},
        )::Cvoid
      elseif $elty <: Complex
        @ccall $lname.$fname2(
          which::Ref{UInt8},
          jobu::Ref{UInt8},
          jobv::Ref{UInt8},
          m::Ref{Int},
          n::Ref{Int},
          kmax::Ref{Int},
          p::Ref{Int},
          k::Ref{Int},
          maxiter::Ref{Int},
          aprod::Ptr{Nothing},
          U::Ptr{$elty},
          ldu::Ref{Int},
          s::Ptr{$elty},
          bnd::Ptr{$subty},
          V::Ptr{$elty},
          ldv::Ref{Int},
          tolin::Ref{$subty},
          work::Ptr{$subty},
          lwork::Ref{Int},
          cwork::Ptr{$elty},
          lcwork::Ref{Int},
          iwork::Ptr{Int},
          liwork::Ref{Int},
          doption::Ptr{$subty},
          ioption::Ptr{Int},
          info::Ptr{Int},
          dparm::Ptr{$elty},
          iparm::Ptr{Int},
        )::Cvoid
      end

      info[1] == 0 || error("lansvd_irl return code: $(info[1])")

      return U, s, V, bnd
    end

    function lansvd_irl(
      which::Char,
      jobu::Char,
      jobv::Char,
      m::Integer,
      n::Integer,
      kmax::Integer,
      p::Integer,
      k::Integer,
      maxiter::Integer,
      pff::Ptr{Nothing},
      initvec::Vector{$elty},
      tolin::$subty,
      dparm::Ptr{Nothing},
    )

      # Extract
      U = Matrix{$elty}(undef, m, kmax + 1)
      copyto!(U, 1, initvec, 1, m)
      s = Vector{$subty}(undef, k)
      bnd = Vector{$subty}(undef, k)
      V = Matrix{$elty}(undef, n, kmax)

      nb = 16 # BLAS-3 blocking size. Don't know the size. It's almost surely a power of 2.
      if jobu == 'N' && jobv == 'N'
        lwork = Int(m + n + 10kmax + 2kmax * kmax + 5 + max(m, max(n, 4kmax + 4)))
        liwork = Int(2kmax + 1)
      else
        lwork =
          Int(m + n + 10kmax + 5kmax * kmax + 4 + max(3kmax * kmax + 4kmax + 4, nb * max(m, n)))
        liwork = Int(8kmax)
      end
      work = Vector{$subty}(undef, lwork)
      iwork = Vector{Int}(undef, liwork)
      if $elty <: Complex
        lcwork = Int(m + n + 32m)
        cwork = Vector{$elty}(undef, lcwork)
      else
        cwork = $elty[]
      end

      ϵ = eps($subty)
      doption = $subty[sqrt(ϵ); ϵ^(3 / 4); 0.0; ϵ^(1 / 6)]
      ioption = Int[0; 1]

      iparm = Int[0]

      (U, s, V, bnd) = lansvd_irl!(
        which,
        jobu,
        jobv,
        m,
        n,
        kmax,
        p,
        maxiter,
        pff,
        U,
        s,
        bnd,
        V,
        tolin,
        work,
        iwork,
        doption,
        ioption,
        dparm,
        iparm,
        cwork = cwork,
      )
      return (U[:, 1:k], s, V[:, 1:k], bnd)
    end
  end
end
