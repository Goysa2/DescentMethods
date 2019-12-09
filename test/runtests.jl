using DescentMethods
@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

using NLPModels
using LinearOperators, LinearAlgebra
using Krylov

using Stopping
using LineSearch
using DescentMethods

function rosenbrock(x)
	n = 2; m = 2;
	f = []
	push!(f, 10 * (x[2]-x[1]^2))
	push!(f, (x[1]-1))
	return sum(f[i]^2 for i=1:m)
end
x₀ = [-1.2, 1.0]

solvers = [:CG_FR, :CG_HZ, :CG_HS, :CG_PR, :NewtonLDLT, :Newton, :Newlbfgs, :Shamanskii]

for solver in solvers
    nlp = ADNLPModel(rosenbrock, x₀)
    println("Testing $(String(solver))")
    nlpatx = NLPAtX(nlp.meta.x0)
    nlpstop = NLPStopping(nlp, Stopping.unconstrained, nlpatx)

    final_nlp_at_x, optimal = eval(solver)(nlp, nlpstop, verbose = true, linesearch = armijo_ls)
    println("optimal = $(string(optimal))")
    finalize(nlp)
end
