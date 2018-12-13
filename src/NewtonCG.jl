export NwtdirectionCG, NewtonCG
"""
Computes the Newton direction using the conjugate gradient iteravtive scheme
"""
function NwtdirectionCG(H,∇f;verbose::Bool=false)
    # printstyled("dans NwtdirectionCG \n", color = :green)
    e=1e-6
    n = length(∇f)
    τ = 0.5 # need parametrization
    cgtol = max(e, min(0.7, 0.01 * norm(∇f)^(1.0 + τ)))

    # cgTN doesn't work properly yet
    # cg (from JuliaSmoothOptimizers/Krylov) seems to do the job...

    (d, cg_stats) = cg(H, -∇f,
                       atol=cgtol, rtol=0.0,
                       itmax=max(2 * n, 50),
                       verbose=verbose)

    return d
end
