using Test
using Random
using LinearAlgebra
using Statistics
using Optimisers

include(joinpath(@__DIR__, "..", "src", "ACEUtils.jl"))
using .ACEUtils

# Synthetic observations only; no research dataset is loaded in this suite.
const A_SYNTHETIC = [1.0 0.0; 0.0 2.0; 1.0 1.0; 2.0 -1.0; -1.0 2.0; 0.5 1.5]
const Y_SYNTHETIC = [0.4, -0.3, 0.8, -0.6, 0.2, 0.9]
const INFO_SYNTHETIC = (E_inds=[1, 2], F_inds=[3, 4, 5, 6], W_E=[1.0, 0.5])

# Permit the two initial finite-value scans, then fail during batch copying to
# exercise the solver's exception path after it disables garbage collection.
struct ReadBudgetMatrix <: AbstractMatrix{Float64}
    data::Matrix{Float64}
    reads::Threads.Atomic{Int}
    limit::Int
end
Base.size(A::ReadBudgetMatrix) = size(A.data)
function Base.getindex(A::ReadBudgetMatrix, i::Int, j::Int)
    count = Threads.atomic_add!(A.reads, 1)
    count < A.limit || error("Injected matrix access failure")
    return A.data[i, j]
end

@testset "Loss definitions and analytic gradient" begin
    theta = [0.2, -0.1]
    A, y, info = A_SYNTHETIC, Y_SYNTHETIC, INFO_SYNTHETIC
    residual = A * theta - y
    expected_E = mean(abs2, residual[info.E_inds] .* info.W_E)
    expected_F = mean(abs2, residual[info.F_inds])
    losses = ACEUtils.component_losses(theta, A, y, info)
    @test losses.L_E ≈ expected_E
    @test losses.L_F ≈ expected_F

    a, b, gamma = 0.25, 0.75, 0.125
    gradient = zeros(2)
    value = ACEUtils.weighted_objective_gradient!(gradient, theta, A, y, info; a, b, gamma)
    @test value ≈ a * expected_E + b * expected_F + gamma * sum(abs2, theta)
    delta = 1e-6
    numerical = similar(theta)
    for j in eachindex(theta)
        plus, minus = copy(theta), copy(theta)
        plus[j] += delta
        minus[j] -= delta
        numerical[j] = (
            ACEUtils.weighted_objective_gradient!(nothing, plus, A, y, info; a, b, gamma) -
            ACEUtils.weighted_objective_gradient!(nothing, minus, A, y, info; a, b, gamma)
        ) / (2 * delta)
    end
    @test gradient ≈ numerical rtol=1e-7 atol=1e-9

    # Rescaling a fixed-lambda loss also rescales its regularizer. A static
    # alpha fit must retain its own gamma instead of reusing that objective.
    alpha = 0.25
    lambda = (1-alpha) / alpha
    static = ACEUtils.weighted_objective_gradient!(nothing, theta, A, y, info;
                                                   a=alpha, b=1-alpha, gamma)
    fixed = ACEUtils.weighted_objective_gradient!(nothing, theta, A, y, info;
                                                  a=1.0, b=lambda, gamma)
    @test static ≈ alpha * fixed + (1-alpha) * gamma * sum(abs2, theta)
    @test !isapprox(static, alpha * fixed; rtol=1e-8)
    @test_throws ArgumentError ACEUtils.weighted_objective_gradient!(nothing, theta, A, y, info; a=-1.0)
end

@testset "Observation validation and matrix orientation" begin
    A, y, info = A_SYNTHETIC, Y_SYNTHETIC, INFO_SYNTHETIC
    @test ACEUtils.validate_observations(A, y, info) === nothing
    @test ACEUtils.validate_observations(permutedims(A), y, info; orientation=:columns) === nothing
    @test_throws DimensionMismatch ACEUtils.validate_observations(A, y, info; orientation=:columns)
    @test_throws ArgumentError ACEUtils.validate_observations(zeros(0, 2), Float64[],
        (E_inds=Int[], F_inds=Int[], W_E=Float64[]))
    @test_throws ArgumentError ACEUtils.validate_observations(A, y,
        (E_inds=[1, 2], F_inds=[2, 3], W_E=[1.0, 1.0]))
    @test_throws ArgumentError ACEUtils.validate_observations(A, y,
        (E_inds=[1, 7], F_inds=[3, 4], W_E=[1.0, 1.0]))
    @test_throws DimensionMismatch ACEUtils.validate_observations(A, y,
        (E_inds=[1, 2], F_inds=[3, 4], W_E=[1.0]))
    @test_throws ArgumentError ACEUtils.validate_observations(A, y,
        (E_inds=[1, 2], F_inds=[3, 4], W_E=[1.0, NaN]))
end

function tiny_solver()
    return ACEUtils.ConstrainedPDSolver(primal_opt_rule=Optimisers.Adam(1e-4),
        dual_opt_rule=Optimisers.Adam(1e-2), max_iter=3, epsilon_F=0.0,
        lambda_F_0=0.0, dual_update_freq=2, initial_learning_rate=1e-4,
        lr_decay_step=0, verbose=false)
end

function tiny_pd(; kwargs...)
    A = permutedims(A_SYNTHETIC)
    return ACEUtils.solve_pd2_hpc_complete1(tiny_solver(), A, Y_SYNTHETIC,
        INFO_SYNTHETIC, A, Y_SYNTHETIC, INFO_SYNTHETIC;
        batch_size_E=1, batch_size_F=2, eval_every=2, kwargs...)
end

@testset "PD batching, history and fixed-multiplier handoff" begin
    Random.seed!(42)
    pd = tiny_pd()
    Random.seed!(42)
    repeated = tiny_pd()
    @test pd["C"] == repeated["C"]
    @test pd["lambda_F"] == repeated["lambda_F"]
    @test pd["iterations"] == 3
    @test pd["stopping_reason"] == "max_iter"
    @test pd["lambda_F"] > 0
    @test all(isfinite, pd["C"])
    history = pd["history"]
    @test all(length(v) == pd["iterations"] + 1 for v in values(history))
    @test history["iteration"] == [1, 2, 3, 3]
    @test history["test_evaluated"] == [true, true, false, true]
    @test history["full_train"] == [false, false, false, true]
    @test history["train_iteration"] == [0, 1, 2, 3]
    @test history["test_iteration"] == [1, 2, 2, 3]
    @test history["full_train_evaluated"] == [true, true, false, true]
    @test isnan(history["full_train_L_F"][3])
    @test history["lambda_F_before"][2] == 0.0
    @test history["lambda_F"][2] > history["lambda_F_before"][2]
    @test isnan(history["lambda_F_before"][end])
    @test issorted(history["elapsed_seconds"])
    @test all(>=(0), history["elapsed_seconds"])
    @test pd["elapsed_seconds"] >= history["elapsed_seconds"][end]
    @test history["lambda_F"][end] == pd["lambda_F"]
    @test history["train_L_E"][1] ≈ Y_SYNTHETIC[1]^2
    @test history["train_L_F"][1] ≈ mean(abs2, Y_SYNTHETIC[3:4])
    final_losses = ACEUtils.component_losses(pd["C"], A_SYNTHETIC, Y_SYNTHETIC, INFO_SYNTHETIC)
    @test history["train_L_E"][end] ≈ final_losses.L_E
    @test history["train_L_F"][end] ≈ final_losses.L_F
    @test history["full_train_L_F"][end] ≈ final_losses.L_F

    initial = copy(pd["C"])
    refined = ACEUtils.solve_only_lbfgs(pd["C"], pd["lambda_F"], A_SYNTHETIC,
        Y_SYNTHETIC, INFO_SYNTHETIC, A_SYNTHETIC, Y_SYNTHETIC, INFO_SYNTHETIC;
        max_iter=3, verbose=false, eval_every=2)
    @test pd["C"] == initial
    @test refined["lambda"] == pd["lambda_F"]
    history_keys = ("history_test_L_E", "history_test_L_F", "history_train_loss", "history_iters")
    @test length(unique(length(refined[key]) for key in history_keys)) == 1
    final_objective = ACEUtils.weighted_objective_gradient!(nothing, refined["theta"],
        A_SYNTHETIC, Y_SYNTHETIC, INFO_SYNTHETIC; a=1.0, b=pd["lambda_F"], gamma=1e-8)
    @test refined["history_train_loss"][end] ≈ final_objective
    @test refined["history_iters"][end] == refined["iterations"]

    static = ACEUtils.solve_static_lbfgs(zeros(2), 0.25, A_SYNTHETIC,
        Y_SYNTHETIC, INFO_SYNTHETIC, A_SYNTHETIC, Y_SYNTHETIC, INFO_SYNTHETIC;
        max_iter=3, verbose=false, eval_every=2)
    @test static["alpha"] == 0.25
    @test static["energy_weight"] == 0.25
    @test static["force_weight"] == 0.75
    @test static["l2_reg"] == 1e-8
end

@testset "PD uses post-update full-force residuals" begin
    info = (E_inds=[1], F_inds=[2, 3], W_E=[1.0])
    A = ones(1, 3)
    y = ones(3)
    solver = ACEUtils.ConstrainedPDSolver(primal_opt_rule=Optimisers.Descent(0.25),
        dual_opt_rule=Optimisers.Descent(0.1), initial_learning_rate=0.25,
        max_iter=1, epsilon_F=0.5, lambda_F_0=0.2, dual_update_freq=1,
        verbose=false)
    result = ACEUtils.solve_pd2_hpc_complete1(solver, A, y, info, A, y, info;
        warmup_steps=0)
    # One primal step crosses the constraint boundary: 1.0 -> 0.16.
    @test result["C"] ≈ [0.6]
    @test result["lambda_F"] ≈ 0.166
    @test result["stopping_reason"] == "target_reached"
    history = result["history"]
    @test history["train_L_F"][1] ≈ 1.0
    @test history["full_train_L_F"][1] ≈ 0.16
    @test history["test_L_E"][1] ≈ 0.16
    @test history["test_L_F"][1] ≈ 0.16
    @test history["lambda_F_before"][1] ≈ 0.2
    @test history["lambda_F"][1] ≈ result["lambda_F"]

    # At step 2 the force batch violates the constraint, but the full set does not.
    A = reshape([1.0, 0.0, 0.0], 1, 3)
    y = [1.0, 0.0, 4.0]
    solver = ACEUtils.ConstrainedPDSolver(primal_opt_rule=Optimisers.Descent(0.1),
        dual_opt_rule=Optimisers.Descent(0.1), initial_learning_rate=0.1,
        max_iter=2, epsilon_F=12.0, lambda_F_0=0.5, dual_update_freq=2,
        verbose=false)
    result = ACEUtils.solve_pd2_hpc_complete1(solver, A, y, info, A, y, info;
        batch_size_E=1, batch_size_F=1, eval_every=3, warmup_steps=100)
    @test result["iterations"] == 2
    @test result["lambda_F"] ≈ 0.1
    @test result["history"]["train_L_F"][2] == 16.0
    @test result["history"]["full_train_L_F"][2] == 8.0
    @test !result["history"]["test_evaluated"][2]
    @test result["history"]["full_train_evaluated"][2]
end

@testset "PD force-target stopping cadence and warmup" begin
    A = reshape([1.0, 0.0, 0.0], 1, 3)
    y = zeros(3)
    info = (E_inds=[1], F_inds=[2, 3], W_E=[1.0])
    solver = ACEUtils.ConstrainedPDSolver(max_iter=7, epsilon_F=0.1,
        dual_update_freq=3, verbose=false)
    result = ACEUtils.solve_pd2_hpc_complete1(solver, A, y, info, A, y, info;
        batch_size_E=1, batch_size_F=1, eval_every=2, warmup_steps=3)
    @test result["iterations"] == 6
    @test result["stopping_reason"] == "target_reached"
    history = result["history"]
    @test all(length(v) == 7 for v in values(history))
    @test history["test_iteration"] == [1, 2, 2, 4, 4, 6, 6]
    @test history["full_train_evaluated"] == [true, true, true, true, false, true, true]
    @test isnan(history["full_train_L_F"][5])
    @test history["full_train_L_F"][6] == 0.0
end

@testset "PD updates are independent of test-evaluation cadence" begin
    A, y, info = permutedims(A_SYNTHETIC), Y_SYNTHETIC, INFO_SYNTHETIC
    solver = ACEUtils.ConstrainedPDSolver(max_iter=5, epsilon_F=0.0,
        dual_update_freq=2, verbose=false)
    Random.seed!(7)
    first_run = ACEUtils.solve_pd2_hpc_complete1(solver, A, y, info, A, y, info;
        batch_size_E=1, batch_size_F=2, eval_every=2)
    Random.seed!(7)
    second_run = ACEUtils.solve_pd2_hpc_complete1(solver, A, y, info, A, y, info;
        batch_size_E=1, batch_size_F=2, eval_every=3)
    @test first_run["C"] == second_run["C"]
    @test first_run["lambda_F"] == second_run["lambda_F"]
end

@testset "PD multiplier projection and initial bounds" begin
    A, y, info = permutedims(A_SYNTHETIC), Y_SYNTHETIC, INFO_SYNTHETIC
    cap = 1e-5
    solver = ACEUtils.ConstrainedPDSolver(primal_opt_rule=Optimisers.Adam(1e-4),
        dual_opt_rule=Optimisers.Adam(1e-2), max_iter=3, epsilon_F=0.0,
        lambda_F_max=cap, dual_update_freq=1, initial_learning_rate=1e-4,
        verbose=false)
    pd = ACEUtils.solve_pd2_hpc_complete1(solver, A, y, info, A, y, info)
    @test pd["lambda_F"] == cap
    @test all(x -> 0 <= x <= cap, pd["history"]["lambda_F"])
    @test pd["history"]["lambda_F"][end] == pd["lambda_F"]
    @test pd["iterations"] == 3
    @test_throws ArgumentError ACEUtils.solve_pd2_hpc_complete1(solver,
        A, y, info, A, y, info; lambda_F_init=2cap)
    @test_throws ArgumentError ACEUtils.ConstrainedPDSolver(lambda_F_max=cap, lambda_F_0=2cap)
    for invalid_cap in (0.0, -1.0, Inf, NaN)
        @test_throws ArgumentError ACEUtils.ConstrainedPDSolver(lambda_F_max=invalid_cap)
    end

    # A satisfied constraint drives the dual step through the lower boundary.
    lower = ACEUtils.ConstrainedPDSolver(max_iter=1, epsilon_F=10.0,
        lambda_F_0=cap, lambda_F_max=cap, dual_update_freq=1, verbose=false)
    pd_lower = ACEUtils.solve_pd2_hpc_complete1(lower, A, y, info, A, y, info)
    @test pd_lower["lambda_F"] == 0.0
end

@testset "PD input errors and GC state" begin
    A, y, info = permutedims(A_SYNTHETIC), Y_SYNTHETIC, INFO_SYNTHETIC
    @test_throws DimensionMismatch ACEUtils.solve_pd2_hpc_complete1(tiny_solver(),
        A_SYNTHETIC, y, info, A_SYNTHETIC, y, info)
    @test_throws ArgumentError ACEUtils.solve_pd2_hpc_complete1(tiny_solver(),
        A, y, info, A, y, info; batch_size_E=3)
    @test_throws ArgumentError ACEUtils.solve_pd2_hpc_complete1(tiny_solver(),
        A, y, info, A, y, info; batch_size_F=5)
    @test_throws ArgumentError ACEUtils.solve_pd2_hpc_complete1(tiny_solver(),
        A, y, info, A, y, info; batch_size_F=0)
    previous = GC.enable(false)
    try
        tiny_pd()
        # GC.enable returns the state that was in effect before this call.
        @test GC.enable(false) == false
        GC.enable(true)
        failing = ReadBudgetMatrix(A, Threads.Atomic{Int}(0), 2 * length(A))
        failure = try
            ACEUtils.solve_pd2_hpc_complete1(tiny_solver(), failing, y, info,
                failing, y, info; batch_size_E=1, batch_size_F=2)
            nothing
        catch exception
            exception
        end
        @test failure !== nothing
        if failure !== nothing
            @test occursin("Injected matrix access failure", sprint(showerror, failure))
        end
        @test GC.enable(true) == true
    finally
        GC.enable(previous)
    end
end

include("data_config_tests.jl")
include("metrics_tests.jl")

include("experiment_tests.jl")
