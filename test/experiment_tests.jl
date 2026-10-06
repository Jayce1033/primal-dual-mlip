@testset "Experiment preflight without assembly" begin
    config = load_config("configs/experiments/static.toml"; overrides=Dict("seed" => 42))
    report = run_scan(config; base_grid=true, check=true)
    @test report["fits"] == 9
    @test first(report["alphas"]) ≈ 0.999
    @test last(report["alphas"]) ≈ 0.001
    @test_throws ArgumentError run_scan(config; check=true)
    refined = ACEUtils.merge_config(config, Dict("scan" => Dict("knee_alphas" => [0.25])))
    @test run_scan(refined; check=true)["fits"] == 10
    @test run_scan(config; solver="lsqr", base_grid=true, check=true)["fits"] == 9
    for bad in (-1.0, Inf, NaN, true)
        invalid = ACEUtils.merge_config(config, Dict("scan" => Dict("weight_reference_E" => bad)))
        @test_throws ArgumentError run_scan(invalid; solver="lsqr", base_grid=true, check=true)
    end
    ready = ACEUtils.merge_config(config, Dict("static" => Dict("alpha" => 0.5)))
    wrong_problem = (input_signature=Dict("model" => Dict("order" => 99)),)
    @test_throws ArgumentError train_model(ready; problem=wrong_problem)
end

@testset "LSQR weight ratio and constant norm" begin
    config = load_config("configs/experiments/static.toml")
    info = (E_inds=[1, 2], F_inds=collect(3:11), W_E=[1.0, 1/sqrt(2)])
    problem = (train_info=info,)
    for alpha in (0.001, 0.5, 0.999)
        we, wf = ACEUtils._scan_weights(alpha, problem, config)
        @test we^2 / wf^2 ≈ alpha * 9 / ((1-alpha)*2)
        @test we^2 * 1.5 + wf^2 * 9 ≈ 900 * 1.5 + 9
    end
end

@testset "Ablation scaling keeps the chosen fixed rates" begin
    config = load_config("configs/experiments/ablation.toml")
    @test_throws ArgumentError ACEUtils.ablation_cases(config; sweep="force_batch")
    configured = ACEUtils.merge_config(config, Dict("ablation" => Dict(
        "batch_scaling" => "primal", "reference_batch_size_F" => 1140,
        "reference_primal_lr" => 1e-4, "reference_dual_lr" => 1e-2)))
    cases = ACEUtils.ablation_cases(configured; sweep="force_batch")
    @test length(cases) == 3
    @test last(cases[end])["pd"]["primal_lr"] ≈ 1e-3
    @test last(cases[end])["pd"]["initial_learning_rate"] ≈ 1e-3
    @test last(cases[end])["pd"]["dual_lr"] ≈ 1e-2
    @test last(cases[end])["pd"]["batch_size_E"] == 117
end
