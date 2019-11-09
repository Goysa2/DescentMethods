export Newton

"""
A globalized Newton algorithm with Line Search
Inputs:
    - An AbstractNLPModel (our problem)
    - An NLPStopping (stopping fonctions/criterion of our problem)
    - (opt) hessian_rep, different way to implement the Hessian. The default
        way is a LinearOperator
Outputs:
    - An NLPAtX, so all the information of the last iterate
    - A boolean true if we reahced an optimal solution, false otherwise
"""
function Newton(nlp             :: AbstractNLPModel,
                 nlp_stop       :: NLPStopping;
                 linesearch     :: Function = armijo_ls,
                 Nwtdirection   :: Function = NwtdirectionLDLT,
                 hessian_rep    :: Function = hessian_dense,
                 kwargs...)

    nlp_at_x = nlp_stop.current_state

    n = nlp.meta.nvar

    xt = copy(nlp.meta.x0)
	T = eltype(xt)
    ∇ft = Array{Float64}(undef, n)

    f = obj(nlp, xt)
    ∇f = grad(nlp, xt)
    OK = update_and_start!(nlp_stop, x = xt, fx = f, gx = ∇f, g0 = ∇f)
    # ∇fNorm = BLAS.nrm2(n, nlp_at_x.gx, 1)
    ∇fNorm = norm(nlp_at_x.gx)
    !OK && State.update!(nlp_at_x, Hx = hessian_rep(nlp, xt))

    iter = 0

    @info log_header([:iter, :f, :nrm_g, :α], [Int64, T, T, T])
	@info log_row(Any[iter, nlp_at_x.fx, ∇fNorm, 0.0])
    β = 0.0
    # d = zeros(nlp_at_x.gx)
    d = zero(nlp_at_x.gx)

    h = LineSearch.LineModel(nlp, xt, d)

    while !OK
        d = Nwtdirection(nlp_at_x.Hx, nlp_at_x.gx)
        # slope = BLAS.dot(n, d, 1, nlp_at_x.gx, 1)
        slope = d' * nlp_at_x.gx

        h = LineSearch.redirect!(h, xt, d)

        ls_at_t = LSAtT(0.0, h₀ = nlp_at_x.fx, g₀ = slope)
        stop_ls = LS_Stopping(h, (x, y) -> armijo(x, y, τ₀ = 1e-09), ls_at_t)
        ls_at_t, good_step_size = linesearch(h, stop_ls; kwargs...)
        # good_step_size || (nlp_stop.meta.stalled_linesearch = true)

        xt = nlp_at_x.x + ls_at_t.x * d
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

		@info log_row(Any[iter, nlp_at_x.fx, ∇fNorm, ls_at_t.x])
    end

    return nlp_stop, nlp_stop.meta.optimal
end
