export NwtdirectionLDLT, NewtonLDLT

# """
# Computes the Newton direction using the LDLᵗ factorization.
# Use the factorization from LDLFactorizations (JuliaSmoothOptimizers)
# """
# function NwtdirectionLDLT(H, g; verbose::Bool=false)
#     LDLT = ldl(H)
#     d = LDLT \ -g
#     return d
# end

function NwtdirectionLDLT(H, g; verbose :: Bool = false)
   T = eltype(H)
   n, = size(g)
   global L = Matrix{T}(undef, n, n)
   global D = Matrix{T}(undef, n, n)
   global pp = Vector{Int}(undef, n)
   global ρ = nothing
   global ncomp = nothing

   try
       (L, D, pp, ρ, ncomp) = ldlt_symm(H, 'r')
   catch
 	println("*******   Problem in LDLt")
       #res = PDataLDLt()
       #res.OK = false
       #return res
       d = NaN * zeros(n)
       return d
   end

    # A[pp,pp] = P*A*P' =  L*D*L'

    if true in isnan.(D)
 	println("*******   Problem in D from LDLt: NaN")
        println(" cond (H) = $(cond(H))")
        #res = PDataLDLt()
        #res.OK = false
        #return res
        d = NaN * zeros(n)
        return d
    end

   X = eigen(D)
   Δ = X.values
   Q =  X.vectors
   ϵ2 =  1.0e-8

   Γ = max.(abs.(Δ), ϵ2)

    # Ad = P' * L * Q * Δ * Q' * L' *P d = -g
    # replace Δ by Γ to ensure positive definiteness
   d̃ = L \ g[pp]
   dtemp = -(Q' * d̃ ./ Γ)
   d̂ = L' \ (Q * (Q' * d̃ ./ Γ))
   d = -d̂[invperm(pp)]

   return d
end


function NewtonLDLT(nlp          :: AbstractNLPModel,
                       nlp_stop     :: NLPStopping;
                       verbose      :: Bool=false,
                       verboseLS    :: Bool = false,
                       kwargs...)
    return  Newton(nlp,
                   nlp_stop;
                   verbose = verbose,
                   verboseLS = verboseLS,
                   Nwtdirection = NwtdirectionLDLT,
                   hessian_rep = hessian_dense,
                   kwargs...)
end
