module TSVD

    using Base.BLAS

    function bidiag{T<:Real}(A::Union(AbstractMatrix{T}, AbstractMatrix{Complex{T}}), u0, k, kmax, tol = 1e-12)

        m, n = size(A)
        local τ
        reorth_b = false

        αs = T[]
        βs = T[]
        U = Vector{T}[]
        V = Vector{T}[]
        νs = T[]
        μs = T[1]
        normA = 2sqrt(norm(A, 1)) # hack! Should probably be changed, but I'm not sure how much is matters

        νmaxs = T[]
        μmaxs = T[]

        μsvec = Vector{T}[]

        u = copy(u0)
        β = norm(u)
        scale!(u, inv(β))
        push!(U, copy(u))
        v = zeros(size(A, 2))

        for j = 1:kmax

            # The v step
            ## apply operator
            Ac_mul_B!(one(T), A, u, -β, v)
            α = norm(v)

            ## run ω recurrence
            reorth_ν = Int[]
            for i = 1:j - 1
                τ = 4eps(T)*(hypot(α,β) + hypot(αs[i], βs[i])) + eps(T)*normA
                ν = βs[i]*μs[i + 1] + αs[i]*μs[i] - β*νs[i]
                ν = (ν + copysign(τ, ν))/α
                if abs(ν) > tol
                    push!(reorth_ν)
                end
                νs[i] = ν
            end
            push!(νs, 1)

            ## reorthogonalize if necessary
            if reorth_b || length(reorth_ν) > 0
                for i = reorth_ν
                    axpy!(-dot(V[i], v), V[i], v)
                    νs[i] = eps(T)
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
                τ = 4eps(T)*(hypot(α,β) + hypot(αs[i], (i == j ? β : βs[i]))) + eps(T)*normA
                μ = αs[i]*νs[i] + (i > 1 ? βs[i - 1]*νs[i-1] : zero(T)) - α*μs[i]
                μ = (μ + copysign(τ, μ))/β
                if abs(μ) > tol
                    push!(reorth_μ, i)
                end
                μs[i] = μ
            end
            push!(μs, 1)

            ## reorthogonalize if necessary
            if reorth_b || length(reorth_μ) > 0
                for i = reorth_μ
                    axpy!(-dot(U[i], u), U[i], u)
                    μs[i] = eps(T)
                end
                β = norm(u)
                reorth_b = !reorth_b
            end

            ## update the rsult vectors
            push!(βs, β)
            scale!(u, inv(β))
            push!(U, copy(u)) # copy to avoid aliasing
        end

        push!(αs, 0)
        αs, βs, U, V
    end

end