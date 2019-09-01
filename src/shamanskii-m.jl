export Shamanskii

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
function Shamanskii(nlp            :: AbstractNLPModel,
                    nlp_stop       :: NLPStopping;
                    linesearch     :: Function = one_step_size,
                    verbose        :: Bool = false,
                    Nwtdirection   :: Function = NwtdirectionCG,
                    Shamdirection  :: Function = ShamanskiiDir,
                    hessian_rep    :: Function = hessian_operator,
                    kwargs...)

    nlp_at_x = nlp_stop.current_state

    n = nlp.meta.nvar

    # xt = Array{Float64}(n)
    xt = copy(nlp.meta.x0)
    ∇ft = Array{Float64}(undef, n)

    f = obj(nlp, xt)
    ∇f = grad(nlp, xt)
    OK = update_and_start!(nlp_stop, x = xt, fx = f, gx = ∇f, g0 = ∇f)
    # ∇fNorm = BLAS.nrm2(n, nlp_at_x.gx, 1)
    ∇fNorm = norm(nlp_at_x.gx)
    !OK && update!(nlp_at_x, Hx = hessian_rep(nlp, xt))

    iter = 0

    verbose && @printf("%4s  %8s  %7s  %8s \n", " iter", "f", "‖∇f‖", "α")
    verbose && @printf("%5d  %9.2e  %8.1e", iter, nlp_at_x.fx, ∇fNorm)
    β = 0.0
    # d = zeros(nlp_at_x.gx)
    d = zero(nlp_at_x.gx)

    h = LineModel(nlp, xt, d)

    while !OK
        xp, d = Shamdirection(nlp, nlp_at_x, xt, Nwtdirection)
        # slope = BLAS.dot(n, d, 1, nlp_at_x.gx, 1)
        slope = d' * nlp_at_x.gx

        # verbose && @printf("  %8.1e", slope)

        h = redirect!(h, xt, d)

        ls_at_t = LSAtT(0.0, h₀ = nlp_at_x.fx, g₀ = slope)
        stop_ls = LS_Stopping(h, (x, y) -> armijo(x, y, τ₀ = 1e-09), ls_at_t)
        verbose && println(" ")
        ls_at_t, good_step_size = linesearch(h, stop_ls; kwargs...)
        good_step_size || (nlp_stop.meta.stalled_linesearch = true)


        xt = xp + ls_at_t.x * d

        ft = obj(nlp, xt); ∇ft = grad(nlp, xt)

        # BLAS.blascopy!(n, nlp_at_x.x, 1, xt, 1)
        # BLAS.axpy!(n, ls_at_t.x, d, 1, xt, 1) #BLAS.axpy!(n, t, d, 1, xt, 1)
        ∇ft = grad!(nlp, xt, ∇ft)

        # Move on.
        OK = update_and_stop!(nlp_stop, x = xt, fx = ft, gx = ∇ft, Hx = hessian_rep(nlp, xt))

        # norm(∇f) bug: https://github.com/JuliaLang/julia/issues/11788
        # ∇fNorm = BLAS.nrm2(n, nlp_at_x.gx, 1)
        ∇fNorm = norm(nlp_at_x.gx)
        iter = iter + 1

        verbose && @printf("%4d  %9.2e  %8.1e %8.2e", iter, nlp_at_x.fx, ∇fNorm, ls_at_t.x)

    end

    verbose && @printf("\n")

    return nlp_stop, nlp_stop.meta.optimal
end
