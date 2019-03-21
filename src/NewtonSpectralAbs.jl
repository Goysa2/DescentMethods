export NwtdirectionSpectral, NewtonSpectralAbs

""" Computes the Newton direction using the spectral factorization"""
function NwtdirectionSpectral(H,g;verbose::Bool=false)
    Δ = fill(1.0, size(g)) #ones(g)
    V = fill(1.0, size(H)) #ones(H)
    try
        Δ, V = eigen(H)
    catch
        Δ, V = eigen(H + Matrix(1.0I, size(H)))
    end
    ϵ2 =  1.0e-8
    Γ = 1.0 ./ max.(abs.(Δ),ϵ2)

    # d = - (V * diagm(Γ) * V') * (g)
    d = - (V * Matrix(Diagonal(Γ)) * V') * (g)
    return d
end

"""
Performs a Newton algorithm using a spectral factorization to compute
the Newton direction. For more information see the doc for Newton.
"""
function NewtonSpectralAbs(nlp :: AbstractNLPModel,
                           nlp_stop :: NLPStopping;
                           verbose :: Bool=false,
                           verboseLS :: Bool = false,
                           kwargs...)
    return  Newton(nlp,
                   nlp_stop;
                   verbose = verbose,
                   verboseLS = verboseLS,
                   Nwtdirection = NwtdirectionSpectral,
                   hessian_rep = hessian_dense,
                   kwargs...)
end
