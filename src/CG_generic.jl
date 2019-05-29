export CG_generic

"""
A non-linear conjugate gradient algorithm.
Can use differents conjugate gradient formulae:
  - Fletcher & Reeves
  - Hager & Zhang
  - Pollack & Ribière
  - Hestenes & Stiefel

  Inputs:
  - An AbstractNLPModel (our problem)
  - An NLPStopping (stopping fonctions/criterion of our problem)
  - (opt) verbose = true will show function value, norm of gradient and the
    inner product of the gradient and the descent direction
  - (opt) hessian_rep, different way to implement the Hessian. The default
    way is a LinearOperator

  Outputs:
  - An NLPAtX, so all the information of the last iterate
  - A boolean true if we reahced an optimal solution, false otherwise
"""
function CG_generic(nlp        :: AbstractNLPModel,
                    nlp_stop   :: NLPStopping;
                    verbose    :: Bool = false,
                    linesearch :: Function = TR_Nwt_ls,
                    CG_formula :: Function = formula_HZ,
                    scaling    :: Bool = true,
                    kwargs...)

    nlp_at_x = nlp_stop.current_state
    x = copy(nlp.meta.x0)
    # @show x
    n = nlp.meta.nvar

    xt = Array{Float64}(undef, n)
    ∇ft = Array{Float64}(undef, n)

    f = obj(nlp, x)
    ∇f = grad(nlp, x)

    iter = 0

    #∇f = grad(nlp, x)
    OK = update_and_start!(nlp_stop, x = x, fx = f, gx = ∇f, g0 = ∇f)
    ∇fNorm = norm(nlp_at_x.gx)

    # verbose && @printf("%4s  %8s  %11s %8s  %4s  %2s \n", "iter", "f", "‖∇f‖", "∇f'd", "bk","scale")
    verbose && @printf("%4s  %8s  %7s  %8s \n", " iter", "f", "‖∇f‖", "α")
    verbose && @printf("%5d  %9.2e  %8.1e", iter, nlp_at_x.fx, ∇fNorm)

    # optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,x,f,∇f)

    β = 0.0
    # d = zeros(∇f)
    d = zero(nlp_at_x.gx)
    scale = 1.0

    h = LineModel(nlp, x, d * scale)

    # OK = true
    # stalled_linesearch = false
    # stalled_ascent_dir = false

    while !OK
        d = - ∇f + β * d
        slope = ∇f⋅d
        if slope > 0.0  # restart with negative gradient
          #stalled_ascent_dir = true
          d = - ∇f
          slope =  ∇f⋅d
        end
        # println(" ")
        # printstyled("d = $d \n", color = :yellow)
        # #else
        # verbose && @printf(" %10.1e", slope * scale)

        h = redirect!(h, x, d * scale)
        ls_at_t = LSAtT(0.0, h₀ = nlp_at_x.fx, g₀ = slope)
        stop_ls = LS_Stopping(h, (x, y) -> armijo(x, y, τ₀ = 1e-09), ls_at_t)
        # verbose && println(" ")
        verbose && println(" ")
        ls_at_t, good_step_size = linesearch(h, stop_ls, LS_Function_Meta())
        # printstyled("α = $(ls_at_t.x) \n", color = :yellow)
        # if linesearch in interfaced_ls_algorithms
        #   ft = obj(nlp, x + (t*scale)*d)
        #   nlp.counters.neval_obj += -1
        # end

        ls_at_t.x *= scale
        # printstyled("x = $(nlp_at_x.x) \n", color = :yellow)
        xt = nlp_at_x.x + ls_at_t.x * d
        # printstyled("xt = $xt \n", color = :yellow)
        ft = obj(nlp, xt);
        ∇ft = grad(nlp, xt);

        ∇ft = grad!(nlp, xt, ∇ft)
        # Move on.
        s = xt - x
        y = ∇ft - ∇f
        β = 0.0
        if (∇ft⋅∇f) < 0.2 * (∇ft⋅∇ft)   # Powell restart
            β = CG_formula(∇f, ∇ft, s, d)
        end
        if scaling
            scale = (y⋅s) / (y⋅y)
        end
        if scale <= 0.0
            #println(" scale = ",scale)
            #println(" ∇f⋅s = ",∇f⋅s,  " ∇ft⋅s = ",∇ft⋅s)
            scale = 1.0
        end
        x = xt
        f = ft
        dxt = xt - x
        BLAS.blascopy!(n, ∇ft, 1, ∇f, 1)

        ∇fNorm = norm(∇ft, Inf)
        # @show ∇fNorm
        iter = iter + 1
        verbose && @printf("%4d  %9.2e  %8.1e %8.2e", iter, nlp_at_x.fx, ∇fNorm, ls_at_t.x)

        OK = update_and_stop!(nlp_stop, x = xt, fx = ft, gx = ∇ft)
        # printstyled("on a x  = $x  \n", color = :yellow)
        # printstyled("on a xt = $xt \n", color = :yellow)
        # printstyled("on a nlp_at_x.x = $(nlp_at_x.x) \n", color = :yellow)
        # printstyled("✔ \n ", color = :yellow )
    end
    verbose && @printf("\n")


    return nlp_at_x, nlp_stop.meta.optimal
end
