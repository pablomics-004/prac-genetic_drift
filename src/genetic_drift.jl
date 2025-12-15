#=
Usage
    julia --project=. ./src/genetic_drift.jl -d 42 -s 1000 -g 100 -m 1000
=#
using StatsBase: sample
using ArgParse
using GLMakie
using Random

const VF64 = Vector{Float64}

function plot_unit(
    Y1::Vector{VF64},
    Y2::Vector{VF64},
    x::AbstractVector{<:Integer}
)::Figure
    f = Figure()
    ax1 = Axis(f[1,1], title="Frequency-generations")
    ax2 = Axis(f[1,2], title="Heterozygosity-generations")

    axs = (ax1, ax2)
    Ys = (Y1, Y2)

    ylbl = ("A1 frequency", "Heterozygosity")
    xlbl = "Generations"

    for (idx, Y) in enumerate(Ys)

        axs[idx].ylabel = ylbl[idx]
        axs[idx].xlabel = xlbl

        for y in Y; lines!(axs[idx], x, y); end
    end

    return f
end

function genetic_drift(
    rng::AbstractRNG,
    size::Int, 
    generations::Int
)::Tuple{Vector{Float64},Vector{Float64}}
    α = 4
    N = size * 2
    invN = 1.0 / N
    # rng = MersenneTwister(seed)

    He = Vector{Float64}(undef, generations)
    counts = zeros(UInt32, N)
    pa_1 = similar(He)
    touched = Int[]
    sizehint!(touched, N)

    alleles = 1:N
    parent_alleles = rand(rng, alleles, N)
    next_alleles = similar(parent_alleles)
    a1 = sample(rng, alleles, 3; replace=false)

    for gen in 1:generations
        empty!(touched)
        # Vector x indices
        @inbounds for i in 1:N
            e = parent_alleles[rand(rng, alleles)]
            next_alleles[i] = e
            
            if counts[e] == 0
                push!(touched, e)
            end

            counts[e] += 1
        end
        pA1 = Float64(sum(counts[idx] for idx in a1)) * invN

        pa_1[gen] = pA1
        He[gen] = 2*pA1*(1 - pA1)

        k = length(touched)
        if α * k ≤ N # k/N ≤ 1/α
            @inbounds for i in touched
                counts[i] = 0
            end
        else
            fill!(counts, 0)
        end

        # Next generation is the current generation
        parent_alleles, next_alleles = next_alleles, parent_alleles
    end
    return pa_1, He
end

function get_args()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--seed", "-d"
            help = "Seed for random experiments"
            arg_type = Int
            default = 42
        "--size", "-s"
            help = "Population size"
            arg_type = Int
            default = 5
        "--generations", "-g"
            help = "Simulated generations"
            arg_type = Int
            default = 1000
        "--simulations", "-m"
            help = "Number of genetic drift simulations"
            arg_type = Int
            default = 1
    end

    return parse_args(s)
end

function main()
    args = get_args()

    pa1 = Vector{VF64}(undef, args["simulations"])
    He = similar(pa1)

    rng = MersenneTwister(args["seed"])

    for i in eachindex(pa1)
        va1, vhe = genetic_drift(rng, args["size"], args["generations"])
        pa1[i] = va1
        He[i] = vhe
    end

    println("Genetic drift computation finished! Plotting in process...")

    generations = 1:args["generations"]
    f = plot_unit(pa1, He, generations)
    display(f)
    wait(f.scene)
end

# Will execute only if the file is run
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end