include(joinpath(@__DIR__, "common.jl"))

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    exit(main("analyze_pareto"))
end
