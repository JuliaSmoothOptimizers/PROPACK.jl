module PROPACK

export tsvd, tsvdvals

include("wrappers.jl")

global __mat__

#Track number of matvecs
global mvp1s = 0
global mvp2s = 0

# function __f__(transa, mm, nn, x, y, dparm, iparm)
#     if char(unsafe_load(transa)) == 'n'
#         A_mul_B!(1.0, __mat__, pointer_to_array(x, unsafe_load(nn)), 0.0, pointer_to_array(y, unsafe_load(mm)))
#     else
#         Ac_mul_B!(1.0, __mat__, pointer_to_array(x, unsafe_load(mm)), 0.0, pointer_to_array(y, unsafe_load(nn)))
#     end
#     nothing
# end
function __f__(transa, mm, nn, x, y, dparm, iparm)
    if char(unsafe_load(transa)) == 'n'
        tmp = __mat__*pointer_to_array(x, unsafe_load(nn))
        unsafe_copy!(y, pointer(tmp), unsafe_load(mm))
        mvp1s += 1
    else
        tmp = __mat__'pointer_to_array(x, unsafe_load(mm))
        unsafe_copy!(y, pointer(tmp), unsafe_load(nn))
        mvp2s += 1
    end
    nothing
end

function tsvd{T}(A; initvec::Vector{T} = zeros(T, size(A, 1)), k::Integer = 1, kmax::Integer = 1000, tolin::Real = sqrt(eps(real(one(T)))))

    m, n = size(A)
    global __mat__ = A
    global mvp1s = 0
    global mvp2s = 0
    __pf__ = cfunction(__f__, Void, (Ptr{UInt8}, Ptr{Int32}, Ptr{Int32}, Ptr{T}, Ptr{T}, Ptr{T}, Ptr{Int32}))

    lansvd('Y', 'Y', m, n, __pf__, initvec, k, kmax, tolin)
end
function tsvdvals{T}(A; initvec::Vector{T} = zeros(T, size(A, 1)), k::Integer = 1, kmax::Integer = 1000, tolin::Real = sqrt(eps(real(one(T)))))

    m, n = size(A)
    global __mat__ = A
    global mvp1s = 0
    global mvp2s = 0
    __pf__ = cfunction(__f__, Void, (Ptr{UInt8}, Ptr{Int32}, Ptr{Int32}, Ptr{T}, Ptr{T}, Ptr{T}, Ptr{Int32}))

    lansvd('N', 'N', m, n, __pf__, initvec, k, kmax, tolin)[2]
end

end # module
