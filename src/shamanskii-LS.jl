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
                 	   linesearch   :: Function = shamanskii_line_search,
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
	nlp_stop.meta.optimality0 = norm(copy(∇f))
    OK = update_and_start!(nlp_stop, x = xt, fx = f, gx = ∇f)#, g0 = ∇f)
    ∇fNorm = norm(nlp_at_x.gx)

	global Hhat = hessian_rep(nlp, xt)
    global iter = 0
	global i = 0
	global fm1 = nothing

	global L = Matrix{T}(undef, n, n)
	global D = Matrix{T}(undef, n, n)
	global pp = Vector{Int}(undef, n)
	global ρ = nothing
	global ncomp = nothing
	global Q = nothing
	global Γ = nothing
	global ϵ2 = nothing

    @info log_header([:iter, :f, :nrm_g, :α], [Int64, T, T, T])
	@info log_row(Any[iter, nlp_at_x.fx, ∇fNorm, 0.0])

    β = 0.0
    d = zero(nlp_at_x.gx)
    h = LineSearch.LineModel(nlp, xt, d)

    while !OK
		if !(iter == i * p) && (u == 0)
			# println("we do nothing")
		else
			# println("we do something")
			(L, D, pp, ρ, ncomp) = ldlt_symm(hessian_rep(nlp, xt), 'r')
			X = eigen(D)
			Δ = X.values
			Q =  X.vectors
			ϵ2 = sqrt(eps(T))
			Γ = max.(abs.(Δ), ϵ2)
			# @show Γ
			# @show Γ .== ϵ2
			if (true in (Δ .== ϵ2))
			   # too_diff = true
			   u = 1
			else
			   # too_diff = false
			   u = 0
			end
			# @show u
			if (iter == i * p)
				i += 1
			end
		end
		# @show L
		# @show D
		# @show Q
		# @show pp
		# @show Γ
		# @show ϵ2
		# @show u
        # d, u = Nwtdirection(Hhat, nlp_at_x.gx)
		g = copy(nlp_at_x.gx)
		# @show eltype(g)
		d̃ = L \ g[pp]
		# @show eltype(d̃)
		d̂ = L' \ (Q * (Q' * d̃ ./ Γ))
		# @show eltype(d̂)
		d = -d̂[invperm(pp)]
		# @show eltype(d)

		# @show u
        slope = d' * nlp_at_x.gx

        h = LineSearch.redirect!(h, xt, d)

        ls_at_t = LSAtT(0.0, h₀ = nlp_at_x.fx, g₀ = slope)
        stop_ls = LS_Stopping(h, (x, y) -> armijo(x, y, τ₀ = 1e-09), ls_at_t)
        ls_at_t, good_step_size = linesearch(h, stop_ls; kwargs...)
        xt  = nlp_at_x.x + ls_at_t.x * d
		fm1 = copy(nlp_at_x.fx)
        ft  = obj(nlp, xt); ∇ft = grad(nlp, xt)

        ∇ft = grad!(nlp, xt, ∇ft)
		# @show eltype(∇ft)
		# @show eltype(xt)
		# @show eltype(ft)
		OK = update_and_stop!(nlp_stop, x = xt, fx = ft, gx = ∇ft)

		# @show ls_at_t.x
		# @show ca
		# @show fm1
		# @show ft
		# @show cf * abs(ft) + fm1
		# @show cf * abs(ft) + fm1 < ft

		if (fm1 < ft)
			u = 1
		else
			u = 0
		end

        # Move on.
        ∇fNorm = norm(nlp_at_x.gx)
        iter = iter + 1

		@info log_row(Any[iter, nlp_at_x.fx, ∇fNorm, ls_at_t.x])
    end

    return nlp_stop, nlp_stop.meta.optimal
end
