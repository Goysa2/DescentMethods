export Halley_dir_AD

function Halley_dir_AD(nlp            :: AbstractNLPModel,
                       nlp_at_x       :: NLPAtX,
                       x              :: AbstractVector,
                       Nwtdirection   :: Function)

    n = length(x)

    dₙ = Nwtdirection(nlp_at_x.Hx, nlp_at_x.gx, verbose = false)

    e₁ = zeros(n)
    e₁[1] = 1.0
    ∇³fdₙ = ∇f³xuv(nlp, x, dₙ, e₁)
    for i = 2:n
        eᵢ = zeros(n)
        eᵢ[i] = 1.0
        ∇³fdₙ = hcat(∇³fdₙ, ∇f³xuv(nlp, x, dₙ, eᵢ))
    end

    hess_plus_tenseur = (nlp_at_x.Hx + 0.5   .* ∇³fdₙ)

    # dH = -(hess_plus_tenseur \ nlp_at_x.gx)
    dH = Nwtdirection(hess_plus_tenseur, nlp_at_x.gx)

    return dH
end
