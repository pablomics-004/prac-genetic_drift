#!/usr/bin/env julia

# ====================== PACKAGES ======================

using Distributions
using CairoMakie
using Random

# ====================== FUNCTIONS ======================

const VF64 = Vector{Float64}

function plot_on_axis(
    ax::Axis,
    Ys::Vector{VF64},
    x::AbstractVector{<:Real};
    labels::Union{Nothing, Vector{String}} = nothing,
    linestyle = :solid,
    xlbl::AbstractString = "",
    ylbl::AbstractString = ""
)
    ax.xlabel = xlbl
    ax.ylabel = ylbl

    if isnothing(labels)
        for y in Ys
            lines!(ax, x, y; linestyle = linestyle)
        end
    else
        for (y, lab) in zip(Ys, labels)
            lines!(ax, x, y; label = lab, linestyle = linestyle)
        end
    end

    return ax
end

function genetic_drift(
    rng::AbstractRNG,
    size::Int = 5,
    p0::Float64 = 0.2,
    H0::Float64 = 0.2,
    generations::Int = 10
)::Tuple{Vector{Float64},Vector{Float64}}

    inv2N = 1.0 / Float64(2 * size)

    pA1 = Vector{Float64}(undef, generations)
    Ht = similar(pA1)

    pA1[1] = p0
    Ht[1] = Ht

    alleles = 2 * size

    @inbounds for g in 2:generations
        pA1[g] = rand(rng, Binomial(alleles, pA1[g-1])) / alleles
        Ht[g] = H0 * (1.0 - inv2N)^(g-1)
    end
    
    return pA1, Ht
end

# ====================== MAIN ======================

function main()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end