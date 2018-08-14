using PROPACK
using Base.Test

ϵ = eps(Float64)

# construct a matrix with chosen singular values
m, n = 10, 5  # choose n ≤ m
σ = collect(n:-1:1)  # in decreasing order because Propack always returns them so
Σ = [diagm(σ) ; zeros(m-n, m-n)]
U, _ = qr(rand(m, m))
V, _ = qr(rand(n, n))
A = U * Σ * V'

# compute a few leading singular values and vectors with lansvd
k = 3
tolin=sqrt(ϵ)
U, s, V, bnd, nprod, ntprod = tsvd(A, k=k, tolin=tolin)
@assert size(U) == (m, k)
@assert length(s) == k
@assert size(V) == (n, k)
@assert length(bnd) == k
println("computed singular values: ", s)
println("exact singular values: ", σ[1:k])
println("error bounds reported: ", bnd)
println("number of matvecs and transposed matvecs: ", nprod, ", ", ntprod)
for i = 1 : k
  @assert bnd[i] ≤ max(16ϵ * s[1], tolin * s[i])
  @assert abs(s[i] - σ[i]) ≤ tolin * σ[i]
end
@assert norm(U * diagm(s) * V' - A) ≤ 1.2 * σ[k+1]

# ensure tsvdvals returns the same singular value estimates
s2, _, nprod, ntprod = tsvdvals(A, k=k, tolin=tolin)
println("computed singular values: ", s2)
println("number of matvecs and transposed matvecs: ", nprod, ", ", ntprod)
@assert norm(s - s2) ≤ sqrt(ϵ) * norm(s)

# compute the two smallest singular values with tsvd_irl
# we use A' to skip zero singular values
k = 2
σs = σ[n-k+1:n]
U, s, V, bnd, nprod, ntprod = tsvd_irl(A', k=k, tolin=tolin)
@assert size(U) == (n, k)
@assert length(s) == k
@assert size(V) == (m, k)
@assert length(bnd) == k
println("computed smallest singular values: ", s)
println("exact smallest singular values: ", σs)
println("error bounds reported: ", bnd)
println("number of matvecs and transposed matvecs: ", nprod, ", ", ntprod)
for i = 1 : k
  @assert bnd[i] ≤ max(16ϵ * s[1], tolin * s[i])
  @assert abs(s[i] - σs[i]) ≤ tolin * σs[i]
end

# ensure tsvdvals_irl returns the same singular value estimates
s2, _, nprod, ntprod = tsvdvals_irl(A', k=k, tolin=tolin)
println("computed smallest singular values: ", s2)
println("number of matvecs and transposed matvecs: ", nprod, ", ", ntprod)
@assert norm(s - s2) ≤ sqrt(ϵ) * norm(s)
