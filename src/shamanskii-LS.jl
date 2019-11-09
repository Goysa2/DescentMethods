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
function Shamanskii_LS(nlp          :: AbstractNLPModel,
                 	   nlp_stop     :: NLPStopping;
                 	   linesearch   :: Function = armijo_ls,
                 	   Nwtdirection :: Function = NwtdirectionLDLT_LS,
                 	   hessian_rep  :: Function = hessian_dense,
					   η 		    :: Float64 = 1.5,
					   ca           :: Float64 = 0.1,
					   cf           :: Float64 = 0.25,
					   u            :: Int = 0,
					   p            :: Int = 1,
                 	   kwargs...)

    nlp_at_x = nlp_stop.current_state
    n = nlp.meta.nvar
    xt = copy(nlp.meta.x0)
	T = eltype(xt)
    ∇ft = Array{Float64}(undef, n)

    f = obj(nlp, xt)
    ∇f = grad(nlp, xt)
    OK = update_and_start!(nlp_stop, x = xt, fx = f, gx = ∇f, g0 = ∇f)
    ∇fNorm = norm(nlp_at_x.gx)

	global Hhat = hessian_rep(nlp, xt)
    global iter = 0
	global i = 0
	global fm1 = nothing

    @info log_header([:iter, :f, :nrm_g, :α], [Int64, T, T, T])
	@info log_row(Any[iter, nlp_at_x.fx, ∇fNorm, 0.0])

    β = 0.0
    d = zero(nlp_at_x.gx)
    h = LineSearch.LineModel(nlp, xt, d)

    while !OK
        d, u = Nwtdirection(Hhat, nlp_at_x.gx)
		@show u
        slope = d' * nlp_at_x.gx

        h = LineSearch.redirect!(h, xt, d)

        ls_at_t = LSAtT(0.0, h₀ = nlp_at_x.fx, g₀ = slope)
        stop_ls = LS_Stopping(h, (x, y) -> armijo(x, y, τ₀ = 1e-09), ls_at_t)
        ls_at_t, good_step_size = linesearch(h, stop_ls; kwargs...)
        xt  = nlp_at_x.x + ls_at_t.x * d
		fm1 = copy(nlp_at_x.fx)
        ft  = obj(nlp, xt); ∇ft = grad(nlp, xt)

        ∇ft = grad!(nlp, xt, ∇ft)
		OK = update_and_stop!(nlp_stop, x = xt, fx = ft, gx = ∇ft)

		# if (ls_at_t.x >= ca) || (((ft - fm1)/ft) >= cf)
		# 	u = 0
		# end

        # Move on.
		@show iter
		@show i * p
		@show u
		@show (iter == i * p)
		@show !(iter == i * p)
		if !(iter == i * p) && (u == 0)
			println("we do nothing")
		else
			println("we do something")
			Hhat = copy(hessian_rep(nlp, xt))
			if (iter == i * p)
				i += 1
			end
		end

        ∇fNorm = norm(nlp_at_x.gx)
        iter = iter + 1

		@info log_row(Any[iter, nlp_at_x.fx, ∇fNorm, ls_at_t.x])
    end

    return nlp_stop, nlp_stop.meta.optimal
end

# function Shamanskii_LS(nlp            :: AbstractNLPModel,
#                        nlp_stop       :: NLPStopping;
#                        linesearch     :: Function = armijo_ls,
#                        verbose        :: Bool = false,
#                        Nwtdirection   :: Function = NwtdirectionLDLt,
#                        hessian_rep    :: Function = hessian_dense,
#                        mem            :: Int = 2,
#                        η              :: Float64 = 1.5, # η > 1
#                        cₐ             :: Float64 = 0.1, # cₐ ∈ (0, 1]
#                        c𝐟              :: Float64 = 0.25, # c𝐟 ∈ (0, 1)
#                        p              :: Int = 10, # p > 1
#                        kwargs...)
#
#     nlp_at_x = nlp_stop.current_state
#
#     L = nothing; Q = nothing; Γ = nothing
#     D = nothing; pp = nothing; rho = nothing; ncomp = nothing;
#     αₐ = 0.0; αₛ = 0.0;
#
#     x = copy(nlp.meta.x0) # our x₀
#     n = nlp.meta.nvar
#
#     # xt = Array{Float64}(n)
#     xₖ = copy(nlp.meta.x0)
#     ∇fₖ = Array{Float64}(undef, n)
#
#     f = obj(nlp, x)
#     fₖ₋₁ = f
#     ∇fₖ = grad(nlp, x)
#
#     # Initiliazation to avoir scope issues
#     Hₖ = nothing; Hₖ₋₁ = nothing; approx_Hₖ = nothing; approx_Hₖ₋₁ = nothing;
#     # Step 0 of the algorithm, initialize some parameters
#     k = 0; i = 0; u = 0
#
#     # Step 1 of the algorithm, we check if the initial point is stationnary
#     OK = update_and_start!(nlp_stop, x = x, fx = f, gx = ∇fₖ, g0 = ∇fₖ)
#
#
#     # ∇fNorm = BLAS.nrm2(n, nlp_at_x.gx, 1)
#     ∇fNorm = norm(nlp_at_x.gx)
#     # update!(nlp_at_x, Hx = hessian_rep(nlp, x))
#     # Hₖ₋₁ = nlp_at_x.Hx
#
#     verbose && @printf("%4s  %8s  %7s  %8s  %5s  %5s  %6s  %6s \n", " k", "f", "‖∇f‖", "k", "i", "p", "αₐ", "αₛ")
#     verbose && @printf("%5d  %9.2e  %8.1e  %5d  %5d  %5d  %7.2e  %7.2e \n", k, nlp_at_x.fx, ∇fNorm, k, i, p, αₐ, αₛ)
#
#     d = fill(0.0, size(nlp_at_x.gx)[1])
#
#
#     h = LineSearch.LineModel(nlp, x, d)
#
#     # Step 1 : We check if we have a stationnary point
#     while !OK #k < 10
#         # Step 2 : We compute the exact hessian
#         if (k == i * p) || (u == 1)   # !((k != i * p) && (u == 0))
#             # println("on est dans le if (k == i * p) || (u == 1)")
#             # Compute Hₖ and construct its positive definite approximation
#             # by applying the modified Cholesky factorization. Then u = 0 or
#             # u = 1 depending on wheter or not ̂Hₖ is to different fom Hₖ.
#             Hₖ = hessian_rep(nlp, xₖ)
#             State.update!(nlp_at_x, Hx = Hₖ)
#
#             (L, D, pp, rho, ncomp) = ldlt_symm(Hₖ)
#             # println("On se rend ici ↓")
#             # @show D
#             X = eigen(D)
#             # println("on a X ↑")
#             Δ = X.values
#             Q =  X.vectors
#
#             ϵ2 =  1.0e-8
#             Γ = max.(abs.(Δ),ϵ2)
#             # sizeD = size(D)[1]^2
#             # for j = 1:sizeD
#             #     dj = D[j]
#             #     D[j] = max(dj, Γ)
#             # end
#
#
#             # println("on check si on a une bonne approximation de la hessienne avec Cholesky modifié")
#             # println("isposdef(Hₖ) = $(isposdef(Hₖ))")
#             good_hess_approx = hess_approx(Hₖ, L, D, η)
#             # println("k = $k and good_hess_approx = $good_hess_approx")
#             if good_hess_approx
#                 u = 0
#             else
#                 u = 1
#             end
#
#             # println("i = $i and p = $p")
#             if (k == i * p)
#                 i += 1
#             end
#         end # if (k == i * p) || (u == 1)
#
#         # Step 3 : ???
#         # ̂Hₖ = ̂Hₖ₋₁
#
#         # Step 4: compute the descent direction and xₖ₊₁
#         # d = Nwtdirection(nlp_at_x.Hx, nlp_at_x.gx, verbose = false)
#         d̃ = L\nlp_at_x.gx[pp]
#         d̂ = L'\ (Q*(Q'*d̃ ./ Γ))
#         d = - d̂[invperm(pp)]
#
#         slope = BLAS.dot(n, d, 1, nlp_at_x.gx, 1)
#
#         # verbose && @printf("  %8.1e", slope)
#
#         h = LineSearch.redirect!(h, xₖ, d)
#
#         ls_at_t = LSAtT(0.0, h₀ = nlp_at_x.fx, g₀ = slope)
#         stop_ls = LS_Stopping(h, (x, y) -> armijo(x, y, τ₀ = 1e-09), ls_at_t)
#         ls_at_t, good_step_size = linesearch(h, stop_ls; kwargs...)
#
#         # ls_at_t = LSAtT(0.0, h₀ = nlp_at_x.fx, g₀ = slope)
#         # ls_at_t1 = LSAtT(0.0, h₀ = nlp_at_x.fx, g₀ = slope)
#         # ls_at_t2 = LSAtT(0.0, h₀ = nlp_at_x.fx, g₀ = slope)
#         # stop_ls1 = LS_Stopping(h, (x, y) -> shamanskii_stop(x, y), ls_at_t1)
#         # stop_ls2 = LS_Stopping(h, (x, y) -> shamanskii_stop(x, y), ls_at_t2)
#         # verbose && println(" ")
#         # # ls_at_t, good_step_size = linesearch(h, stop_ls, LS_Function_Meta())
#         # ls_at_t1, good_step_size1 = shamanskii_line_search(h, stop_ls1, LS_Function_Meta())
#         # ls_at_t2, good_step_size2 = armijo_ls(h, stop_ls2, LS_Function_Meta())
#         # good_step_size1 || (nlp_stop.meta.stalled_linesearch = true)
#
#         αₛ = ls_at_t.x
#         # αₐ = ls_at_t2.x
#         # println("good_step_size = $good_step_size  and αₖ = $αₖ")
#
#         # xₖ = nlp_at_x.x + αₖ * d
#         xₖ = nlp_at_x.x + αₛ * d
#         fₖ = obj(nlp, xₖ); ∇fₖ = grad(nlp, xₖ)
#
#         BLAS.blascopy!(n, nlp_at_x.x, 1, xₖ, 1)
#         BLAS.axpy!(n, ls_at_t.x, d, 1, xₖ, 1) #BLAS.axpy!(n, t, d, 1, xt, 1)
#         ∇fₖ = grad!(nlp, xₖ, ∇fₖ)
#
#         # Step 5: We update the value of u
#         # if (αₖ >= cₐ) || (((fₖ - fₖ₋₁)/abs(fₖ)) >= c𝐟)
#         if (αₛ >= cₐ) || (((fₖ - fₖ₋₁)/abs(fₖ)) >= c𝐟)
#             # println("on a modifié u")
#             u = 0
#             # k = k + 1
#         end
#
#
#         OK = update_and_stop!(nlp_stop, x = xₖ, fx = fₖ, gx = ∇fₖ)
#
#         # norm(∇f) bug: https://github.com/JuliaLang/julia/issues/11788
#         # ∇fNorm = BLAS.nrm2(n, nlp_at_x.gx, 1)
#         ∇fNorm = norm(nlp_at_x.gx)
#         k = k + 1
#
#         verbose && @printf("%5d  %9.2e  %8.1e  %5d  %5d  %5d  %7.2e  %7.2e \n", k, nlp_at_x.fx, ∇fNorm, k, i, p, αₐ, αₛ)
#
#     end
#
#     verbose && @printf("\n")
#
#     return nlp_at_x, nlp_stop.meta.optimal
# end
