export NwtdirectionLDLT, NewtonLDLT

"""
Computes the Newton direction using the LDLᵗ factorization.
Use the factorization from LDLFactorizations (JuliaSmoothOptimizers)
"""
function NwtdirectionLDLT(H,g;verbose::Bool=false)
    LDLT = ldl(H)
    d = LDLT \ -g
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
