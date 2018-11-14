module DescentMethods

using Compat, Printf
using NLPModels
using State, Stopping, LineSearch
using LinearAlgebra
using LinearOperators
using SparseArrays

include("HessianDense.jl")
include("HessianOp.jl")
include("HessianSparse.jl")
include("ldlt_symm.jl")
include("Newton.jl")
include("cgTN.jl")
include("NewtonCG.jl")
include("NewtonLDLtAbs.jl")
include("NewtonSpectralAbs.jl")

include("lbfgs.jl")

include("shamanskii-m.jl")




end # module
