export Newlbfgs

"""
A globalized L-BFGS algorithm with Line Search
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
function Newlbfgs(nlp :: AbstractNLPModel,
                  nlp_stop :: NLPStopping;
                  linesearch :: Function = TR_Nwt_ls,
                  verbose   :: Bool=false,
                  verboseLS :: Bool = false,
                  mem       :: Int=5,
                  kwargs...)

    nlp_at_x = nlp_stop.current_state
    x = copy(nlp.meta.x0)
    n = nlp.meta.nvar

    xt = copy(nlp.meta.x0)
    ∇f = grad(nlp, nlp.meta.x0)  # à changer pour 0.7

    f = obj(nlp, x)
    # g = grad(nlp, x) # Nécessaire??
    H = InverseLBFGSOperator(n, mem, scaling=true)
    update!(nlp_at_x, x = x, fx = f, gx = ∇f, Hx = H)
    OK = start!(nlp_stop)
    iter = 0
    ∇fNorm = BLAS.nrm2(n, nlp_at_x.gx, 1)

    verbose && @printf("%4s  %8s  %7s  %8s\n", "iter", "f", "‖∇f‖", "∇f'd")
    verbose && @printf("%4d  %8.1e  %7.1e", iter, f, ∇fNorm)


    d = zeros(nlp_at_x.gx)
    h = LineModel(nlp, x, d)

    while !OK
        d = - H * ∇f
        slope = BLAS.dot(n, d, 1, nlp_at_x.gx, 1)
        if slope > 0.0 # qu'est ce qu'on fait maintenant?
          verbose && @printf("  %8.1e", slope)
        else
          # Perform linesearch.
          h = redirect!(h, xt, d)

          ls_at_t = LSAtT(0.0, h₀ = nlp_at_x.fx, g₀ = slope)
          stop_ls = LS_Stopping(h, (x, y) -> armijo(x, y, τ₀ = 0.01), ls_at_t)
          verbose && println(" ")
          ls_at_t, good_step_size = linesearch(h, stop_ls, LS_Function_Meta())

          good_step_size || (nlp_stop.meta.stalled_linesearch = true)

          xt = nlp_at_x.x + ls_at_t.x * d
          ft = obj(nlp, xt); ∇ft = grad(nlp, xt)
          BLAS.blascopy!(n, nlp_at_x.x, 1, xt, 1) # BLAS.blascopy!(n, x, 1, xt, 1)
          BLAS.axpy!(n, ls_at_t.x, d, 1, xt, 1) # BLAS.axpy!(n, t, d, 1, xt, 1)
          ∇ft = grad!(nlp, xt, ∇ft)

          # Update L-BFGS approximation.
          push!(H, ls_at_t.x * d, ∇ft - ∇f)

          # Move on.
          x = xt
          f = ft

          update!(nlp_at_x, x = xt, fx = ft, Hx = H)

          BLAS.blascopy!(n, ∇ft, 1, ∇f, 1)
          # norm(∇f) bug: https://github.com/JuliaLang/julia/issues/11788
          ∇fNorm = BLAS.nrm2(n, ∇f, 1)
          iter = iter + 1

          verbose && @printf("%4d  %8.1e  %7.1e", iter, f, ∇fNorm)
        end
        OK = stop!(nlp_stop)
    end
    verbose && @printf("\n")


    return nlp_at_x, nlp_stop.meta.optimal
end
