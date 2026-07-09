#!/usr/bin/env julia

#=

Gillespie, J. H. (2004). Population genetics: A concise guide (2nd ed.). The Johns Hopkins University Press.
=#

# ====================== PACKAGES ======================

using Distributions
using CairoMakie
using ArgParse
using Random

# ====================== FUNCTIONS ======================

const VF64 = Vector{Float64}

function plot_on_axis(
    ax::Axis,
    Ys::Vector{Vector{Float64}}, # Asumiendo tu vector de vectores
    x::AbstractVector{<:Real};
    labels::Union{Nothing, Vector{String}} = nothing,
    linestyle = :solid,
    xlbl::AbstractString = "",
    ylbl::AbstractString = "",
    cmap = :viridis
)
    ax.xlabel = xlbl
    ax.ylabel = ylbl

    num_lineas = length(Ys)
    colores = cgrad(cmap, num_lineas, categorical=true)

    if isnothing(labels)
        for (i, y) in enumerate(Ys)
            lines!(ax, x, y; linestyle = linestyle, color = colores[i])
        end
    else
        for (i, (y, lab)) in enumerate(zip(Ys, labels))
            lines!(ax, x, y; label = lab, linestyle = linestyle, color = colores[i])
        end
    end

    return ax
end

function genetic_drift(
    rng::AbstractRNG,
    size::Int = 5,
    p0::Float64 = 0.2,
    generations::Int = 10
)::Vector{Float64}
    #=
    Genetic drift simulation for a diploid hermaphroditic and panmictic population
    according to the second Chapter of the Gillespie, 2004, p. 46-56.

    Parameters
    ----------
        rng: seed for reproducibility.
        size: initial population size.
        p0: initial allelic frequency.

    Return
    -------
        pA1: vector of allele frequencies
    =#

    pA1 = Vector{Float64}(undef, generations)

    pA1[1] = p0

    # Genetic pool
    alleles = 2 * size

    # Run simulations
    @inbounds for g in 2:generations
        pA1[g] = rand(rng, Binomial(alleles, pA1[g-1])) / alleles
    end
    
    return pA1
end

function genetic_drift_zygosity(
    size::Int = 5,
    H0::Float64 = 0.2,
    generations::Int = 10
)::Tuple{Vector{Float64},Vector{Float64}}
    #=
    Impact of the genetic drift simulation for a diploid hermaphroditic and 
    panmictic population according to the second Chapter of the Gillespie, 2004, p. 46-56.
    Because this computation mainly depends in the population size and the initial
    heterozigosity no random generator (seed) is used.

    Parameters
    ----------
        size: initial population size.
        H0: initial heterozigosity.
        generations: number of generations to simulate.

    Returns
    -------
        Ht: vector of heterocigosity frequencies.
        Gt: vector of homocigosity frequencies.
    =#
    
    inv2N = 1.0 / Float64(2 * size)

    Ht = Vector{Float64}(undef, generations)
    Gt = similar(Ht)

    Ht[1] = H0
    Gt[1] = 1 - H0

    # Run simulations
    @inbounds for g in 2:generations
        Ht[g] = H0 * (1.0 - inv2N)^(g-1)
        Gt[g] = 1 - Ht[g]
    end
    
    return Ht, Gt
end

function argument_parser()
    s = ArgParseSettings(
        description = "Parameters for running and saving images of the Wright-Fisher model."
    )

    @add_arg_table! s begin
        "--size", "-i"
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
        "--heterozigosity", "-t"
            help = "Initial heterozigosity"
            arg_type = Float64
            default = 0.5
        "--generations", "-g"
            help = "Number of generations for each genetic drift simulation"
            arg_type = Int
            default = 100
        "--seed"
            help = "Random seed generator using Xoshiro256++ algorithm"
            arg_type = Int
            default = 42
        "--output_dir", "-o"
            help = "Output directory for generated images"
            arg_type = String
            default = "."
        "--allele_image", "-a"
            help = "File name for the allele frequency evolution simulation image"
            arg_type = String
            default = "genetic-drift_allelefreq"
        "--zigosity_image", "-z"
            help = "File name for the allele homozigosity and heterozigosity simulation image"
            arg_type = String
            default = "genetic-drift_hetero-homo"
        "--file_format", "-f"
            help = "File format for image files from the following: jpg, png, jpeg, svg"
            arg_type = String
            default = "png"
        "--pixels", "-x"
            help = "Pixels per unit (definition of the image)"
            arg_type = Int
            default = 2
    end

    return parse_args(s)
end

# ====================== MAIN ======================

function main(
    size::Int, p0::Float64, seed::Int,
    H0::Float64, sims::Int, gens::Int,
    outdir::String, format::String, 
    allele_file::String, zigosity_file::String,
    pixels::Int
)

    # ====================== GENETIC DRIFT SIMULATION ======================

    @info "Memory allocation in the main function."

    # Memory allocation
    pA1 = Vector{Vector{Float64}}(undef, sims)
    generations = collect(1:gens)

    # Random generator
    rng = Xoshiro(seed)

    @info "Starting genetic drift simulation..."

    for i in eachindex(pA1)
        pA1[i] = genetic_drift(rng, size, p0, gens)
    end

    @info "Frequency successfully computed!"

    @info "Starting heterozygosity and homozygosity computation..."
    hetero, homo = genetic_drift_zygosity(size, H0, gens)

    @info "Heterozygosity and homozygosity successfully computed!"

    # ====================== FREQUENCY PLOTS ======================

    @info "Beginning to plot for allele frequency..."

    f = Figure()
    ax = Axis(f[1,1:2])

    plot_on_axis(ax, pA1, generations; xlbl="Generations", ylbl="Allelic frequency (p)", cmap=:inferno)

    # Saving image
    allele_file = "$(allele_file).$format"
    format ∈ Set(["png", "jpg", "jpeg"]) ? 
        save(joinpath(outdir, allele_file), f, px_per_unit=pixels) : 
        save(joinpath(outdir, allele_file), f)

    @info "Plot saved at $allele_file"

    # ====================== ZYGOSITY PLOTS ======================

    @info "Beginning to plot hetero-/homozygosity..."
    g = Figure()
    ax = Axis(g[1,1])

    Labels = ["Heterozygosity", "Homozygosity"]
    plot_on_axis(ax, [hetero, homo], generations; labels=Labels, xlbl="Generations", ylbl="Frequency", cmap=:Set1)
    Legend(g[1,2], ax)

    # Saving image
    zigosity_file = "$(zigosity_file).$format"
    format ∈ Set(["png", "jpg", "jpeg"]) ? 
        save(joinpath(outdir, zigosity_file), g, px_per_unit=pixels) : 
        save(joinpath(outdir, zigosity_file), g)
    
    @info "Plot saved at $zigosity_file"

    # ====================== COMBINED PLOTS ======================

    @info "Beginning to plot allele frequency and zygosity..."

    f = Figure(size = (1200, 500))

    ax1 = Axis(f[1, 3:4], title="Hetero-/homozigosity", titlefont=:bold)
    Labels = ["Heterozygosity", "Homozygosity"]
    plot_on_axis(ax1, [hetero, homo], generations; labels=Labels, cmap=:Set1)
    axislegend(ax1, position = :lt)

    ax2 = Axis(f[1, 1:2], title="Allele frequency (p)", titlefont=:bold)
    plot_on_axis(ax2, pA1, generations; cmap=:inferno)

    Label(f[0, :], "Genetic drift simulation", font = :bold, fontsize=18)
    Label(f[1:2, 0], "Frequency", font = :bold, rotation = pi/2, fontsize=18)
    Label(f[2, :], "Generations", font = :bold, fontsize=18)

    combined_file = "genetic_drift_combined-julia.$format"
    
    if format ∈ Set(["png", "jpg", "jpeg"])
        save(joinpath(outdir, combined_file), f, px_per_unit=pixels)
    else
        save(joinpath(outdir, combined_file), f)
    end

    @info "Combined plot successfully saved at $combined_file"
end

if abspath(PROGRAM_FILE) == @__FILE__

    @info "Starting script execution..."
    
    # ========================= PARAMETERS =========================
    args = argument_parser()

    # Genetic drift simulation parameters
    size = Int(args["size"])
    seed = Int(args["seed"])
    pA = Float64(args["allele_freq"])
    H = Float64(args["heterozigosity"])
    simulations = Int(args["simulations"])
    generations = Int(args["generations"])

    # File and directory managment
    outdir = args["output_dir"]
    file_format = lowercase(args["file_format"])

    if file_format ∉ Set(["png", "svg","jpg", "jpeg"])
        @warn "Invalid file format '$file_format'. 'png' will be used."
        file_format = "png"
    end

    pixels = Int(args["pixels"])
    allele_file = args["allele_image"]
    zigosity_file = args["zigosity_image"]

    mkpath(outdir)
    
    # ========================= MAIN =========================
    main(size, pA, seed, H, simulations, generations, outdir, file_format, allele_file, zigosity_file, pixels)

    @info "Script finished successfully!"
end