export hessian_sparse
"""
Returns the hessian in a sparse format.
"""
function hessian_sparse(nlp,x)
    n = length(x)
    H=hess(nlp,x)
    tempH = (H+tril(H,-1)')
    H = tempH
    return H
end
