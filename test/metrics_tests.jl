# Included by runtests.jl after ACEUtils has been loaded.
function pareto_row(alpha, energy_loss, force_loss;
                    energy_rmse=sqrt(energy_loss), force_rmse=sqrt(force_loss))
    return Dict("alpha" => alpha, "test_L_E" => energy_loss, "test_L_F" => force_loss,
        "energy_rmse_meV_per_atom" => energy_rmse,
        "force_rmse_meV_per_A" => force_rmse)
end

@testset "Physical RMSE differs from optimization normalization" begin
    A = [1.0 0.0; 0.0 1.0; 1.0 1.0; -1.0 2.0]
    y = [0.4, -0.3, 0.8, -0.6]
    theta = zeros(2)
    info = (E_inds=[1, 2], F_inds=[3, 4], W_E=[1.0, 0.5], N_atoms_E=[1, 4])
    metrics = ACEUtils.evaluate_fit(theta, A, y, info)
    @test metrics["L_E"] ≈ mean(abs2, [0.4, -0.15])
    @test metrics["energy_rmse_meV_per_atom"] ≈ 1000 * sqrt(mean(abs2, [0.4, -0.075]))
    @test metrics["force_rmse_meV_per_A"] ≈ 1000 * sqrt(mean(abs2, [0.8, -0.6]))
    @test !isapprox(metrics["energy_rmse_meV_per_atom"], 1000 * sqrt(metrics["L_E"]))
    fallback = ACEUtils.evaluate_fit(theta, A, y,
        (E_inds=info.E_inds, F_inds=info.F_inds, W_E=info.W_E))
    @test fallback == metrics
end

@testset "Nondominated selection and two knee definitions" begin
    rows = [pareto_row(0.1, 1.0, 100.0), pareto_row(0.5, 2.0, 2.0),
            pareto_row(0.9, 100.0, 1.0), pareto_row(0.7, 3.0, 3.0)]
    result = ACEUtils.analyze_pareto(rows)
    @test result["nondominated_indices"] == [1, 2, 3]
    @test result["nondominated_alphas"] == [0.1, 0.5, 0.9]
    @test result["log_loss_knee"]["status"] == "unique"
    @test result["log_loss_knee"]["alphas"] == [0.5]
    @test result["linear_rmse_knee"]["status"] == "unique"
    @test result["linear_rmse_knee"]["alphas"] == [0.5]

    tied = [pareto_row(0.1, 1.0, 100.0), pareto_row(0.4, 2.0, 2.0),
            pareto_row(0.6, 2.0, 2.0), pareto_row(0.9, 100.0, 1.0)]
    result = ACEUtils.analyze_pareto(tied)
    @test result["log_loss_knee"]["status"] == "tied"
    @test result["log_loss_knee"]["alphas"] == [0.4, 0.6]
    @test result["linear_rmse_knee"]["status"] == "tied"

    chord = [pareto_row(0.1, 1.0, 100.0), pareto_row(0.5, 10.0, 10.0),
             pareto_row(0.9, 100.0, 1.0)]
    result = ACEUtils.analyze_pareto(chord)
    @test result["log_loss_knee"]["status"] == "no_utopia_facing_point"
    @test isempty(result["log_loss_knee"]["alphas"])
    @test result["linear_rmse_knee"]["alphas"] == [0.5]
end

@testset "Degenerate paths and invalid Pareto coordinates" begin
    one = ACEUtils.analyze_pareto([pareto_row(0.5, 2.0, 2.0)])
    @test one["log_loss_knee"]["status"] == "degenerate_range"
    @test one["linear_rmse_knee"]["status"] == "degenerate_range"

    zero_loss = [pareto_row(0.1, 0.0, 1.0), pareto_row(0.9, 1.0, 0.0)]
    result = ACEUtils.analyze_pareto(zero_loss)
    @test result["log_loss_knee"]["status"] == "nonpositive_loss"
    @test result["linear_rmse_knee"]["status"] == "tied"
    @test_throws ArgumentError ACEUtils.analyze_pareto(Dict[])
    @test_throws ArgumentError ACEUtils.analyze_pareto([
        pareto_row(0.5, -1.0, 1.0; energy_rmse=1.0)])
    @test_throws ArgumentError ACEUtils.analyze_pareto([pareto_row(0.5, NaN, 1.0)])
    @test_throws ArgumentError ACEUtils.analyze_pareto([
        pareto_row(0.5, 1.0, 1.0; force_rmse=Inf)])
end
