module ACEUtils

using ACEfit, ACEpotentials, AtomsBase, AtomsCalculators
using BSON, ExtXYZ, JSON, Optim, Optimisers
using Dates, LinearAlgebra, Printf, Random, SHA, Statistics, TOML, UUIDs
using Unitful: ustrip

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

include("config.jl")
include("data.jl")
include("models.jl")
include("objectives.jl")
include("solvers/primal_dual.jl")
include("solvers/lbfgs.jl")
include("metrics.jl")
include("train_model.jl")
include("experiments.jl")

export load_config, validate_config, load_datasets, build_model, get_data_indices
export prepare_problem, train_model, run_scan, run_ablation, run_warmstart
export ConstrainedPDSolver, solve_pd2_hpc_complete1, solve_only_lbfgs, solve_static_lbfgs
export component_losses, evaluate_fit, analyze_pareto

end
