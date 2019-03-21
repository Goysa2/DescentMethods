export SHalley_dir_AD

function SHalley_dir_AD(nlp            :: AbstractNLPModel,
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

    hess_plus_tenseur = (nlp_at_x.Hx + ∇³fdₙ)

    ∇³fdₙdₙ = ∇f³xuv(nlp, x, dₙ, dₙ)

    dV = Nwtdirection(hess_plus_tenseur, nlp_at_x.gx)

    grad_plus_tuv = -(nlp_at_x.gx + 0.5 .* ∇f³xuv(nlp, x, dₙ, dV))

    dSH = Nwtdirection(nlp_at_x.Hx, -grad_plus_tuv)

    return dSH
end
