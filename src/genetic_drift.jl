#!/usr/bin/env julia

#=

Gillespie, J. H. (2004). Population genetics: A concise guide (2nd ed.). The Johns Hopkins University Press.
=#

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
)::Tuple{Vector{Float64},Vector{Float64},Vector{Float64}}
    #=
    Genetic drift simulation for a diploid hermaphroditic and panmictic population
    according to the second Chapter of the Gillespie, 2004, p. 46-56.

    Parameters
    ----------
        rng: seed for reproducibility.
        size: initial population size.
        p0: initial allelic frequency.
        H0: initial heterozigosity.

    Returns
    -------
        pA1: vector of allele frequencies.
        Ht: vector of heterocigosity frequencies.
        Gt: vector of homocigosity frequencies.
    =#
    
    inv2N = 1.0 / Float64(2 * size)

    pA1 = Vector{Float64}(undef, generations)
    Ht = similar(pA1)
    Gt = similar(pA1)

    pA1[1] = p0
    Ht[1] = H0
    Gt[1] = 1 - H0

    # Genetic pool
    alleles = 2 * size

    # Run simulations
    @inbounds for g in 2:generations
        pA1[g] = rand(rng, Binomial(alleles, pA1[g-1])) / alleles
        Ht[g] = H0 * (1.0 - inv2N)^(g-1)
        Gt[g] = 1 - Ht[g]
    end
    
    return pA1, Ht, Gt
end

function argument_parser()
    s = ArgParseSettings(
        description = "Parameters for running and saving images of the Wright-Fisher model."
    )

    @add_arg_table! s begin
        "--size", "-z"
            help = "Initial population size for the genetic drift simulation"
            arg_type = Int
            default = 5
        "--simulations", "-s"
            help = "Number of simulations to be runned"
            arg_type = Int
            default = 100
        "--allele_freq", "-p"
            help = "Initial allele frequency"
            arg_type = Float64
            default = 0.5
        "--heterozigosity", "-h"
            help = "Initial heterozigosity"
            arg_type = Float64
            default = 0.5
        "--output_dir", "-o"
            help = "Output directory for generated images"
            arg_type = String
            default = "."
        "--allele_image", "-a"
            help = "File name for the allele frequency evolution simulation image"
            arg_type = String
            default = "genetic-drift_allelefreq"
        "--heterohomo_image", "-h"
            help = "File name for the allele homozigosity and heterozigosity simulation image"
            default = "genetic-drift_hetero-homo"
        "--file_format", "-f"
            help = ""
    end

    return parse_args(s)
end

# ====================== MAIN ======================

function main()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end