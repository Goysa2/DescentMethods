export hessian_operator
"""
Return the value of the hessian of a multivariate function as a LinearOperator.
For more information on LinearOperators, see the doc for the package of the
same name.
"""
function hessian_operator(nlp,x)
    n = nlp.meta.nvar
    # temp = Array(Float64, n)
    # temp = Array{Float64}(n)
    temp = Array{Float64}(undef, n)
    return hess_op!(nlp, x, temp)
    #return LinearOperator(nlp.meta.nvar, nlp.meta.nvar, true, true, v -> hprod(nlp,x,v))
end
