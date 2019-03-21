function hess_approx(h, L, D, η; ϵ = sqrt(eps()))
    n = size(D)[1]
    θ = Vector{Float64}(undef, n)
    ω = Vector{Float64}(undef, n)
    ν = Vector{Float64}(undef, n)

    θ[n] = 0
    ν[1] = h[1, 1]

    # ξ = maximum(abs.(LDLt.L))
    ξ = maximum(abs.(UnitLowerTriangular(h)- Matrix(Diagonal(ones(n)))))
    γ = maximum(abs.(diag(h)))
    β² = max(ϵ, γ, ξ/n)
    β = sqrt(β²)
    # println("ξ/n = $(ξ/n) γ = $γ β = $β")

    ω[1] = max(β, abs(ν[1]), (θ[1]/β)^2)

    good_approx = true; j = 1

    # println("avant le while")

    while (good_approx) && (j <= n)
        # println("dans le while j = $j")
        if j > 1
            ν[j] = h[j, j] - sum(L[j, s]^2 * ω[s] for s = 1:j-1)
        end
        # println("ν[j] = $(ν[j])")

        for i=j+1:n
            # println("dans le for i = $i")
            # println(" θ[j] = $(θ[j])")
            # println(" maximum( abs.( h[i,j])) = $(maximum( abs.( h[i,j])))")
            if j > 1
                # println("on est dans le if qui fait la somme")
                sum_ljs_lis = sum(L[j, s] * L[i, s] * ω[s] for s=1:j-1)
                # println("sum_ljs_lis = $sum_ljs_lis")
            else
                # println("on est dans le if qui ne fait pas la somme")
                sum_ljs_lis = 0.0
                # println("sum_ljs_lis = $sum_ljs_lis")
            end
            θ[j] = max(θ[j],maximum( abs.( h[i,j] .- sum_ljs_lis)))
        end


        if (θ[j]/β)^2 >= η * ν[j]
            good_approx = false
        else
            good_approx = true
        end
        # println("j = $j good_approx = $good_approx")
        # println("ν[j] = $(ν[j]) θ[j] = $(θ[j])")
        j += 1
    end

    return good_approx
end
