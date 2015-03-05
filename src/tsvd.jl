module TSVD

    using Base.BLAS

    import Base.LinAlg: A_mul_B!, Ac_mul_B!, BlasComplex, BlasFloat, BlasReal

    A_mul_B!{T<:BlasFloat}(α::T, A::StridedMatrix{T}, x::StridedVector{T}, β::T, y::StridedVector{T}) = gemv!('N', α, A, x, β, y)
    Ac_mul_B!{T<:BlasReal}(α::T, A::StridedMatrix{T}, x::StridedVector{T}, β::T, y::StridedVector{T}) = gemv!('T', α, A, x, β, y)
    Ac_mul_B!{T<:BlasComplex}(α::T, A::StridedMatrix{T}, x::StridedVector{T}, β::T, y::StridedVector{T}) = gemv!('C', α, A, x, β, y)

    function bidiag{T<:Real}(A::Union(AbstractMatrix{T}, AbstractMatrix{Complex{T}}), steps, initVec, αs = Array(T, 0), βs = Array(T, 0), U = Array(Vector{T}, 0), V = Array(Vector{T}, 0), μs = ones(T, 1), νs = Array(T, 0), reorth_in = false; tolError = 1e-12)

        m, n = size(A)
        τ = eps(T)*countnz(A)/mean(size(A))*norm(A, 1)
        reorth_b = reorth_in
        nReorth = 0
        nReorthVecs = 0

        maxνs = T[]
        maxμs = T[]

        iter = length(αs)

        if iter == 0
            u = copy(initVec)
            v = zeros(size(A, 2))
            β = norm(u)
            scale!(u, inv(β))
            push!(U, copy(u))
        else
            u = copy(U[iter + 1])
            v = copy(V[iter])
            β = βs[iter]
        end

        for j = iter + (1:steps)

            # The v step
            ## apply operator
            Ac_mul_B!(one(T), A, u, -β, v)
            α = norm(v)

            ## run ω recurrence
            reorth_ν = Int[]
            for i = 1:j - 1
                # τ = eps(T)*(hypot(α, β) + hypot(αs[i], βs[i])) + eps(T)*normA ### this doesn't seem to be better than fixed τ = eps
                ν = βs[i]*μs[i + 1] + αs[i]*μs[i] - β*νs[i]
                ν = (ν + copysign(τ, ν))/α
                if abs(ν) > tolError
                    push!(reorth_ν, i)
                end
                νs[i] = ν
            end
            if j > 1
                push!(maxνs, maximum(abs(νs)))
            end
            push!(νs, 1)

            ## reorthogonalize if necessary
            if reorth_b || length(reorth_ν) > 0
                for i = reorth_ν
                    axpy!(-dot(V[i], v), V[i], v)
                    νs[i] = 2eps(T)
                    nReorthVecs += 1
                end
                α = norm(v)
                reorth_b = !reorth_b
            end

            ## update the rsult vectors
            push!(αs, α)
            scale!(v, inv(α))
            push!(V, copy(v)) # copy to avoid aliasing

            # The u step
            ## apply operator
            A_mul_B!(one(T), A, v, -α, u)
            β = norm(u)

            ## run ω recurrence
            reorth_μ = Int[]
            for i = 1:j
                # τ = eps(T)*(hypot(α, β) + hypot(αs[i], (i == j ? β : βs[i]))) + eps(T)*normA ### this doesn't seem to be better than fixed τ = eps
                μ = αs[i]*νs[i] + (i > 1 ? βs[i - 1]*νs[i-1] : zero(T)) - α*μs[i]
                μ = (μ + copysign(τ, μ))/β
                if abs(μ) > tolError
                    push!(reorth_μ, i)
                end
                μs[i] = μ
            end
            push!(maxμs, maximum(μs))
            push!(μs, 1)

            ## reorthogonalize if necessary
            if reorth_b || length(reorth_μ) > 0
                for i = reorth_μ
                    axpy!(-dot(U[i], u), U[i], u)
                    μs[i] = 2eps(T)
                    nReorthVecs += 1
                end
                β = norm(u)
                reorth_b = !reorth_b
                nReorth += 1
            end

            ## update the result vectors
            push!(βs, β)
            scale!(u, inv(β))
            push!(U, copy(u)) # copy to avoid aliasing
        end

        αs, βs, U, V, μs, νs, reorth_b, maxμs, maxνs, nReorth, nReorthVecs
    end

    function tsvd(A, nVals, maxIter, initVec = randn(size(A,1)); tolConv = 1e-12, tolError = 1e-12)

        cc = max(5, nVals)

        αs, βs, U, V, μs, νs, reorth_b, _ = bidiag(A, cc, initVec, tolError = tolError)
        vals0 = svdvals(Bidiagonal([αs; 0], βs, false))
        vals1 = vals0

        hasConv = false
        while cc <= maxIter
            _, _, _, _, _, _, reorth_b, _ = bidiag(A, 2, initVec, αs, βs, U, V, μs, νs, reorth_b, tolError = tolError)
            vals1 = svdvals(Bidiagonal([αs; 0], βs, false))
            if vals0[nVals]*(1 - tolConv) < vals1[nVals] < vals0[nVals]*(1 + tolConv)
                hasConv = true
                break
            end
            vals0 = vals1
            cc += 2
        end
        if !hasConv
            warn("no convergence")
        end
        U, vals1[1:nVals], V
    end
end