export Cheb_dir_AD

function Cheb_dir_AD(nlp            :: AbstractNLPModel,
                     nlp_at_x       :: NLPAtX,
                     x              :: AbstractVector,
                     Nwtdirection   :: Function)

    dₙ = Nwtdirection(nlp_at_x.Hx, nlp_at_x.gx, verbose = false)

    ∇³fdₙdₙ = ∇f³xuv(nlp, x, dₙ, dₙ)

    dCp = Nwtdirection(nlp_at_x.Hx, ∇³fdₙdₙ, verbose = false)

    dC = dₙ + 0.5 * dCp

    return dC
end
