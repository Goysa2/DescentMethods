export NwtdirectionLDLt, NewtonLDLtAbs

# extremely slow compared to other way to compute Newton direction
# probably bunchkaufman factorization or the computing of eigenvalues?


"""Computes the Newton direction using the LDLᵗ factorization"""
function NwtdirectionLDLt(H,g;verbose::Bool=false)
    # L = Array{Float64}(2)
    # D = Array{Float64}(2)
    global L = Array{Float64}(undef, 2)
    global D = Array{Float64}(undef, 2)
    global pp = Array{Int}(undef, 1)
    global ρ = Float64
    global ncomp = Int64

    try
        (L, D, pp, rho, ncomp) = ldlt_symm(H,'p')
        # LBL = bunchkaufman(H)
    catch
 	println("*******   Problem in LDLt")
        #res = PDataLDLt()
        #res.OK = false
        #return res
        return (NaN, NaN, NaN, Inf, false, true, :fail)
    end
    # LBL = bunchkaufman(Symmetric(H, :L))
    # D = LBL.D
    # L = LBL.L
    # pp = LBL.p

    # A[pp,pp] = P*A*P' =  L*D*L'

    if true in isnan.(D)
 	println("*******   Problem in D from LDLt: NaN")
        println(" cond (H) = $(cond(H))")
        #res = PDataLDLt()
        #res.OK = false
        #return res
        return (NaN, NaN, NaN, Inf, false, true, :fail)
    end

    # Δ, Q = eig(D)
    Δ, Q = eigen(Symmetric(D))

    ϵ2 =  1.0e-8
    Γ = max.(abs.(Δ),ϵ2)

    # Ad = P'*L*Q*Δ*Q'*L'*Pd =    -g
    # replace Δ by Γ to ensure positive definiteness
    d̃ = L\g[pp]
    d̂ = L'\ (Q*(Q'*d̃ ./ Γ))
    d = - d̂[invperm(pp)]

    return d
end


function NewtonLDLtAbs(nlp          :: AbstractNLPModel,
                       nlp_stop     :: NLPStopping;
                       verbose      :: Bool=false,
                       verboseLS    :: Bool = false,
                       kwargs...)
    return  Newton(nlp,
                   nlp_stop;
                   verbose = verbose,
                   verboseLS = verboseLS,
                   Nwtdirection = NwtdirectionLDLt,
                   hessian_rep = hessian_dense,
                   kwargs...)
end
