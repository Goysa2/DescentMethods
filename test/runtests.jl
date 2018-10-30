using DescentMethods
@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

using NLPModels, CUTEst
using LinearOperators

using State
using Stopping
using LineSearch
using DescentMethods


nlp = CUTEstModel("ARWHEAD")

solvers = [:Newton, :NewtonLDLtAbs, :NewtonSpectralAbs, :Newlbfgs]
solvers = [:Newton, :Newlbfgs]
for solver in solvers
    println("$(String(solver))")
    nlpatx = NLPAtX(nlp.meta.x0)
    nlpstop = NLPStopping(nlp, Stopping.unconstrained, nlpatx)

    final_nlp_at_x, optimal = eval(solver)(nlp, nlpstop, verbose = true)

    @test optimal
end
