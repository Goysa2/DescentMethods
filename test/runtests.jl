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


# nlp = CUTEstModel("ARWHEAD")

# solvers = [:Newton, :NewtonLDLtAbs, :NewtonSpectralAbs, :Newlbfgs]
solvers = [:Shamanskii, :Newton, :Newlbfgs]

for solver in solvers
    nlp = CUTEstModel("ARWHEAD")
    println("$(String(solver))")
    nlpatx = NLPAtX(nlp.meta.x0)
    nlpstop = NLPStopping(nlp, Stopping.unconstrained, nlpatx)

    final_nlp_at_x, optimal = eval(solver)(nlp, nlpstop, verbose = true)
    finalize(nlp)
end
