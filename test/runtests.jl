using DescentMethods
@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

using NLPModels, CUTEst
using LinearOperators, LinearAlgebra
using Krylov

using State
using Stopping
using LineSearch
using DescentMethods

solvers = [:CG_FR, :CG_HZ, :CG_HS, :CG_PR, :NewtonLDLT, :Newton, :Newlbfgs, :Shamanskii]

for solver in solvers
    nlp = CUTEstModel("ROSENBR")
    println("Testing $(String(solver))")
    nlpatx = NLPAtX(nlp.meta.x0)
    nlpstop = NLPStopping(nlp, Stopping.unconstrained, nlpatx)

    final_nlp_at_x, optimal = eval(solver)(nlp, nlpstop, verbose = true, linesearch = armijo_ls)
    println("optimal = $(string(optimal))")
    finalize(nlp)
end
