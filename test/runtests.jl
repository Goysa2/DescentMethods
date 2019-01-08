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

solvers = [:Shamanskii_LS, :NewtonLDLtAbs, :NewtonLDLT, :Newton, :Newlbfgs, :Shamanskii]

for solver in solvers
    nlp = CUTEstModel("ARWHEAD")
    println("$(String(solver))")
    nlpatx = NLPAtX(nlp.meta.x0)
    nlpstop = NLPStopping(nlp, Stopping.unconstrained, nlpatx)

    final_nlp_at_x, optimal = eval(solver)(nlp, nlpstop, verbose = true)
    println("optimal = $(string(optimal))")
    finalize(nlp)
end
