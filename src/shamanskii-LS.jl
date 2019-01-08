export Shamanskii_LS

"""
A globalized Shamanskii algorithm with Line Search.
This code is an implementation of the idea presented in:

    Global Convergence Technique for the Newton
        Method with Periodic Hessian Evaluation

By F. LAMPARIELLO and M. SCIANDRONE

JOURNAL OF OPTIMIZATION THEORY AND APPLICATIONS:
Vol. 111, No. 2, pp. 341–358, November 2001
"""
function Shamanskii_LS(nlp            :: AbstractNLPModel,
                       nlp_stop       :: NLPStopping;
                       linesearch     :: Function = shamanskii_line_search,
                       verbose        :: Bool = false,
                       Nwtdirection   :: Function = NwtdirectionLDLT,
                       hessian_rep    :: Function = hessian_dense,
                       mem            :: Int = 2,
                       η              :: Float64 = 1.5, # η > 1
                       cₐ             :: Float64 = 0.1, # cₐ ∈ (0, 1]
                       c𝐟              :: Float64 = 0.25, # c𝐟 ∈ (0, 1)
                       p              :: Int = 10, # p > 1
                       kwargs...)

    # Data of the algorithm
    # η = 1.5 # η > 1
    # cₐ = 0.1 # cₐ ∈ (0, 1]
    # c𝐟 = 0.25 # c𝐟 ∈ (0, 1)
    # p = 10 # p > 1

    nlp_at_x = nlp_stop.current_state

    x = copy(nlp.meta.x0) # our x₀
    n = nlp.meta.nvar

    # xt = Array{Float64}(n)
    xₖ = copy(nlp.meta.x0)
    ∇fₖ = Array{Float64}(undef, n)

    f = obj(nlp, x)
    fₖ₋₁ = f
    ∇fₖ = grad(nlp, x)

    # Initiliazation to avoir scope issues
    Hₖ = nothing; Hₖ₋₁ = nothing; approx_Hₖ = nothing; approx_Hₖ₋₁ = nothing;
    # Step 0 of the algorithm, initialize some parameters
    k = 0; i = 0; u = 0

    # Step 1 of the algorithm, we check if the initial point is stationnary
    OK = update_and_start!(nlp_stop, x = x, fx = f, gx = ∇fₖ, g0 = ∇fₖ)


    ∇fNorm = BLAS.nrm2(n, nlp_at_x.gx, 1)
    # update!(nlp_at_x, Hx = hessian_rep(nlp, x))
    # Hₖ₋₁ = nlp_at_x.Hx

    verbose && @printf("%4s  %8s  %7s  %8s  \n", " k", "f", "‖∇f‖", "∇f'd")
    verbose && @printf("%5d  %9.2e  %8.1e", k, nlp_at_x.fx, ∇fNorm)

    d = fill(0.0, size(nlp_at_x.gx)[1])


    h = LineModel(nlp, x, d)

    # Step 1 : We check if we have a stationnary point
    while !OK #k < 10
        # Step 2 : We compute the exact hessian
        if (k == i * p) || (u == 1)   # !((k != i * p) && (u == 0))
            # Compute Hₖ and construct its positive definite approximation
            # by applying the modified Cholesky factorization. Then u = 0 or
            # u = 1 depending on wheter or not ̂Hₖ is to different fom Hₖ.
            Hₖ = hessian_rep(nlp, x)
            approx_Hₖ = ldl(Hₖ)
            println("on check si on a une bonne approximation de la hessienne avec Cholesky modifié")
            println("isposfed(Hₖ) = $(isposfed(Hₖ))")
            good_hess_approx = hess_approx(Hₖ, approx_Hₖ, η)
            println("k = $k and good_hess_approx = $good_hess_approx")
            if good_hess_approx
                u = 0
            else
                u = 1
            end

            println("i = $i and p = $p")
            if (k == i * p)
                i += 1
            end
        end

        # Step 3 : ???
        # ̂Hₖ = ̂Hₖ₋₁

        # Step 4: compute the descent direction and xₖ₊₁
        # d = Nwtdirection(nlp_at_x.Hx, nlp_at_x.gx, verbose = false)
        d = approx_Hₖ \ -nlp_at_x.gx

        slope = BLAS.dot(n, d, 1, nlp_at_x.gx, 1)

        verbose && @printf("  %8.1e", slope)

        h = redirect!(h, xₖ, d)

        ls_at_t = LSAtT(0.0, h₀ = nlp_at_x.fx, g₀ = slope)
        stop_ls = LS_Stopping(h, (x, y) -> shamanskii_stop(x, y), ls_at_t)
        verbose && println(" ")
        ls_at_t, good_step_size = linesearch(h, stop_ls, LS_Function_Meta())
        good_step_size || (nlp_stop.meta.stalled_linesearch = true)

        αₖ = ls_at_t.x
        println("good_step_size = $good_step_size  and αₖ = $αₖ")

        xₖ = nlp_at_x.x + αₖ * d
        fₖ = obj(nlp, xₖ); ∇fₖ = grad(nlp, xₖ)

        BLAS.blascopy!(n, nlp_at_x.x, 1, xₖ, 1)
        BLAS.axpy!(n, ls_at_t.x, d, 1, xₖ, 1) #BLAS.axpy!(n, t, d, 1, xt, 1)
        ∇fₖ = grad!(nlp, xₖ, ∇fₖ)

        # Step 5: We update the value of u
        if (αₖ >= cₐ) || (((fₖ - fₖ₋₁)/abs(fₖ)) >= c𝐟)
            println("on a modifié u")
            u = 0
            # k = k + 1
        end


        OK = update_and_stop!(nlp_stop, x = xₖ, fx = fₖ)
        BLAS.blascopy!(n, ∇fₖ, 1, nlp_at_x.gx, 1)

        # norm(∇f) bug: https://github.com/JuliaLang/julia/issues/11788
        ∇fNorm = BLAS.nrm2(n, nlp_at_x.gx, 1)
        k = k + 1

        verbose && @printf("%4d  %9.2e  %8.1e ", k, nlp_at_x.fx, ∇fNorm)

    end

    verbose && @printf("\n")

    return nlp_at_x, nlp_stop.meta.optimal
end
