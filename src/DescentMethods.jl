module DescentMethods

using Compat, Printf
using NLPModels
using Stopping, LineSearch
using LinearAlgebra
using LinearOperators
using Krylov
using HSL
using LDLFactorizations
using SparseArrays
using SolverTools, Logging

include("HessianDense.jl")
include("HessianOp.jl")
include("HessianSparse.jl")
include("ldlt_symm.jl")
include("Newton.jl")
include("cgTN.jl")
include("NewtonCG.jl")
include("NewtonLDLT.jl")

include("formulae.jl")
include("CG_generic.jl")
include("CG_FR.jl")
include("CG_PR.jl")
include("CG_HS.jl")
include("CG_HZ.jl")

include("hess_approx.jl")
include("shamanskii-LS.jl")
include("shamanskii-LS-MA57.jl")

include("Cheb-dir-AD.jl")
include("Chebyshev.jl")

include("Halley-dir-AD.jl")
include("Halley.jl")

include("SHalley-dir-AD.jl")
include("SuperHalley.jl")

include("NewtonLDLtAbs.jl")
include("NewtonSpectralAbs.jl")

include("lbfgs.jl")

include("ShamanskiiDir.jl")
include("shamanskii-m.jl")

include("autodiff_high_order_model.jl")


end # module
