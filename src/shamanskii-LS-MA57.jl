export Shamanskii_LS_MA57

"""
A globalized Shamanskii algorithm with Line Search.
This code is an implementation of the idea presented in:

    Global Convergence Technique for the Newton
        Method with Periodic Hessian Evaluation

By F. LAMPARIELLO and M. SCIANDRONE

JOURNAL OF OPTIMIZATION THEORY AND APPLICATIONS:
Vol. 111, No. 2, pp. 341–358, November 2001
"""
function Shamanskii_LS_MA57(nlp          :: AbstractNLPModel,
                 	   		nlp_stop     :: NLPStopping;
                 	   		linesearch   :: Function = shamanskii_line_search,
                 	   		Nwtdirection :: Function = NwtdirectionLDLT_LS,
                 	   		hessian_rep  :: Function = hessian_sparse,
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

	global M = Ma57
	global L = SparseMatrixCSC{Float64, Int64}
	global D = SparseMatrixCSC{Float64, Int64}
	global pp = Array{Int64, 1}
	global s = Array{Float64}
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
			H57 = convert(SparseMatrixCSC{Float64, Int64}, hessian_rep(nlp, xt))
			printstyled("on a H57 \n", color = :yellow)
			try
		        M = Ma57(H57)#, print_level = -1)
		        ma57_factorize(M)
		    catch
		 	println("*******   Problem in MA57_0")
		        M = Ma57(H57,print_level = -1)
		        ma57_factorize(M)
		        res = PDataMA57()
		        res.OK = false
		        return res
		    end
			printstyled("on a M \n", color = :yellow)
			try
		        (L, D, s, pp) = ma57_get_factors(M)
		    catch
		        println("*******   Problem after MA57_1")
		        # println(" Cond(H) = $(cond(full(H)))")
		        res = PDataMA57_0()
		        res.OK = false
		        return res
		    end
			printstyled("on a L, D, s, pp \n", color = :yellow)
			#################  Future object BlockDiag operator?
		    vD1 = diag(D)       # create internal representation for block diagonal D
		    vD2 = diag(D, 1)     #
		    vQ1 = ones(length(vD1))       # vector representation of orthogonal matrix Q
		    vQ2 = zeros(length(vD2))      #
		    vQ2m = zeros(length(vD2))     #
		    veig = copy(vD1)      # vector of eigenvalues of D, initialized to diagonal of D
		                          # if D diagonal, nothing more will be computed
			printstyled("on a vD1, vD2, vQ1, vQ2, vQ2m, veig \n", color = :yellow)
		    i = 1;
		    while i < length(vD1)
		        if vD2[i] == 0.0
		            i += 1
		        else
		            mA = [vD1[i] vD2[i]; vD2[i] vD1[i + 1]] #  2X2 submatrix
		            # DiagmA, Qma = eig(mA)                   #  spectral decomposition of mA
		            X = eigen(mA)
		            DiagmA = X.values
		            Qma = X.vectors
		            veig[i] = DiagmA[1]
		            vQ1[i] = Qma[1, 1]
		            vQ2[i] = Qma[1, 2]
		            vQ2m[i] = Qma[2, 1]
		            vQ1[i + 1] = Qma[2, 2]
		            veig[i + 1] = DiagmA[2]
		            i += 2
		        end
		    end

		    Q = sparse(SparseArrays.spdiagm(0 => vQ1, 1 => vQ2m, -1 => vQ2))
			printstyled("on a Q \n", color = :yellow)
		    Δ = veig
			printstyled("on a  Δ \n", color = :yellow)
			ϵ2 = sqrt(eps(T))
			printstyled("on a  ϵ2 \n", color = :yellow)
			Γ = max.(abs.(Δ), ϵ2)
			printstyled("on a  Γ \n", color = :yellow)
			@show Γ
			@show Γ .== ϵ2
			if (true in (Δ .== ϵ2))
			   u = 1
			else
			   u = 0
			end
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
		@show u
        # d, u = Nwtdirection(Hhat, nlp_at_x.gx)
		g = copy(nlp_at_x.gx)
		@show eltype(g)
		@show L
		@show g
		@show g[pp]
		d̃ = L \ g[pp]
		@show eltype(d̃)
		d̂ = L' \ (Q * (Q' * d̃ ./ Γ))
		@show eltype(d̂)
		d = -d̂[invperm(pp)]
		@show eltype(d)

		# @show u
        slope = d' * nlp_at_x.gx

        h = LineSearch.redirect!(h, xt, d)
		printstyled("on a  h \n", color = :yellow)
        ls_at_t = LSAtT(0.0, h₀ = nlp_at_x.fx, g₀ = slope)
		printstyled("on a  ls_at_t \n", color = :yellow)
        stop_ls = LS_Stopping(h, (x, y) -> armijo(x, y, τ₀ = 1e-09), ls_at_t)
		printstyled("on a  stop_ls \n", color = :yellow)
        ls_at_t, good_step_size = linesearch(h, stop_ls; kwargs...)
		printstyled("on a  fini le line search \n", color = :yellow)
        xt  = nlp_at_x.x + ls_at_t.x * d
		printstyled("on a  xt \n", color = :yellow)
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
