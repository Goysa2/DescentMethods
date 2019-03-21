export ShamanskiiDir

function ShamanskiiDir(nlp            :: AbstractNLPModel,
                       nlp_at_x       :: NLPAtX,
                       x              :: AbstractVector,
                       Nwtdirection   :: Function)

    dₙ = Nwtdirection(nlp_at_x.Hx, nlp_at_x.gx, verbose = false)

    xₚ = x + dₙ

    dS = Nwtdirection(nlp_at_x.Hx, grad(nlp, xₚ), verbose = false)

    return xₚ, dS
end
