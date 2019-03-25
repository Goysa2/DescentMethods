export hessian_dense
"""Compute the full hessian of an nlp (AbstractNLPModel). Should only be used
for lower dimensions problems"""
function hessian_dense(nlp,x)
    n = length(x)
    H=hess(nlp,x)
    tempH = (H+tril(H,-1)')
    H = Matrix(Symmetric(tempH, :L))
    return H
end
