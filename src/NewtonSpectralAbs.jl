export NwtdirectionSpectral, NewtonSpectralAbs

""" Computes the Newton direction using the spectral factorization"""
function NwtdirectionSpectral(H,g;verbose::Bool=false)
    Δ = ones(g)
    V = ones(H)
    try
        Δ, V = eig(H)
    catch
        Δ, V = eig(H + eye(H))
    end
    ϵ2 =  1.0e-8
    Γ = 1.0 ./ max.(abs.(Δ),ϵ2)

    d = - (V * diagm(Γ) * V') * (g)
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
                   nlp_stopl;
                   verbose = verbose,
                   verboseLS = verboseLS,
                   Nwtdirection = NwtdirectionSpectral,
                   hessian_rep = hessian_dense,
                   kwargs...)
end
