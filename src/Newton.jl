export Newton

"""
A globalized Newton algorithm with Line Search
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
function Newton(nlp            :: AbstractNLPModel,
                 nlp_stop       :: NLPStopping;
                 linesearch     :: Function = TR_Nwt_ls,
                 verbose        :: Bool = false,
                 Nwtdirection   :: Function = NwtdirectionCG,
                 hessian_rep    :: Function = hessian_operator,
                 kwargs...)

    nlp_at_x = nlp_stop.current_state

    x = copy(nlp.meta.x0)
    n = nlp.meta.nvar

    # xt = Array{Float64}(n)
    xt = copy(nlp.meta.x0)
    ∇ft = Array{Float64}(undef, n)

    f = obj(nlp, x)
    ∇f = grad(nlp, x)
    # update!(nlp_at_x, x = x, fx = f, gx = ∇f, g0 = ∇f)
    OK = update_and_start!(nlp_stop, x = x, fx = f, gx = ∇f, g0 = ∇f)
    ∇fNorm = BLAS.nrm2(n, nlp_at_x.gx, 1)
    !OK && update!(nlp_at_x, Hx = hessian_rep(nlp, x))

    iter = 0

    verbose && @printf("%4s  %8s  %7s  %8s  \n", " iter", "f", "‖∇f‖", "∇f'd")
    verbose && @printf("%5d  %9.2e  %8.1e", iter, nlp_at_x.fx, ∇fNorm)
    β = 0.0
    # d = zeros(nlp_at_x.gx)
    d = zero(nlp_at_x.gx)
    scale = 1.0

    h = LineModel(nlp, x, d)

    while !OK
        d = Nwtdirection(nlp_at_x.Hx, nlp_at_x.gx, verbose = false)
        slope = BLAS.dot(n, d, 1, nlp_at_x.gx, 1)

        verbose && @printf("  %8.1e", slope)

        h = redirect!(h, xt, d)

        ls_at_t = LSAtT(0.0, h₀ = nlp_at_x.fx, g₀ = slope)
        stop_ls = LS_Stopping(h, (x, y) -> armijo(x, y, τ₀ = 0.01), ls_at_t)
        verbose && println(" ")
        ls_at_t, good_step_size = linesearch(h, stop_ls, LS_Function_Meta())
        good_step_size || (nlp_stop.meta.stalled_linesearch = true)

        xt = nlp_at_x.x + ls_at_t.x * d
        ft = obj(nlp, xt); ∇ft = grad(nlp, xt)

        BLAS.blascopy!(n, nlp_at_x.x, 1, xt, 1)
        BLAS.axpy!(n, ls_at_t.x, d, 1, xt, 1) #BLAS.axpy!(n, t, d, 1, xt, 1)
        ∇ft = grad!(nlp, xt, ∇ft)

        # Move on.
        s = xt - nlp_at_x.x
        y = ∇ft - ∇f
        β = (∇ft⋅y) / (∇f⋅∇f)
        # x = xt
        # f = ft
        OK = update_and_stop!(nlp_stop, x = xt, fx = ft, Hx = hessian_rep(nlp, xt))
        # H = hessian_rep(nlp,x)

        BLAS.blascopy!(n, ∇ft, 1, nlp_at_x.gx, 1)

        # norm(∇f) bug: https://github.com/JuliaLang/julia/issues/11788
        ∇fNorm = BLAS.nrm2(n, nlp_at_x.gx, 1)
        iter = iter + 1

        verbose && @printf("%4d  %9.2e  %8.1e ", iter, nlp_at_x.fx, ∇fNorm)

    end

    verbose && @printf("\n")

    return nlp_at_x, nlp_stop.meta.optimal
end
