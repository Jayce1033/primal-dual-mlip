"""Report normalized losses and physical errors from inputs in eV and angstrom."""
function evaluate_fit(theta, A, y, info)
    losses = component_losses(theta, A, y, info)
    all(isfinite, values(losses)) || throw(ArgumentError("Fitted losses are non-finite."))
    residual = A * theta - y
    atom_counts = hasproperty(info, :N_atoms_E) ? info.N_atoms_E : 1.0 ./ info.W_E.^2
    return Dict("L_E" => losses.L_E, "L_F" => losses.L_F,
        "energy_rmse_meV_per_atom" => 1000 * sqrt(mean(abs2, residual[info.E_inds] ./ atom_counts)),
        "force_rmse_meV_per_A" => 1000 * sqrt(mean(abs2, residual[info.F_inds])))
end

"""Analyze a sampled path without assigning a knee to a degenerate front."""
function analyze_pareto(rows::AbstractVector)
    isempty(rows) && throw(ArgumentError("The sampled path is empty."))
    energies = Float64[r["test_L_E"] for r in rows]
    forces = Float64[r["test_L_F"] for r in rows]
    all(isfinite, energies) && all(isfinite, forces) || throw(ArgumentError("Pareto coordinates must be finite."))
    all(>=(0), energies) && all(>=(0), forces) || throw(ArgumentError("Pareto losses must be nonnegative."))
    nd = [i for i in eachindex(rows) if !any(j ->
        energies[j] <= energies[i] && forces[j] <= forces[i] &&
        (energies[j] < energies[i] || forces[j] < forces[i]), eachindex(rows))]
    sort!(nd; by=i -> (energies[i], forces[i], Float64(rows[i]["alpha"])))
    report = Dict{String,Any}("nondominated_indices" => nd,
        "nondominated_alphas" => [rows[i]["alpha"] for i in nd],
        "log_loss_knee" => Dict("status" => "unavailable", "alphas" => Any[]),
        "linear_rmse_knee" => Dict("status" => "unavailable", "alphas" => Any[]))
    function normalize_pair(x, y)
        dx, dy = maximum(x) - minimum(x), maximum(y) - minimum(y)
        (dx > 0 && dy > 0) || return nothing
        return ((x .- minimum(x)) ./ dx, (y .- minimum(y)) ./ dy)
    end
    if all(>(0), energies[nd]) && all(>(0), forces[nd])
        xy = normalize_pair(log10.(energies[nd]), log10.(forces[nd]))
        if isnothing(xy)
            report["log_loss_knee"]["status"] = "degenerate_range"
        else
            scores = 1 .- xy[1] .- xy[2]
            eligible = findall(>(0), scores)
            if isempty(eligible)
                report["log_loss_knee"]["status"] = "no_utopia_facing_point"
            else
                best = maximum(scores[eligible])
                chosen = [i for i in eligible if isapprox(scores[i], best; rtol=1e-12, atol=1e-14)]
                report["log_loss_knee"] = Dict("status" => length(chosen) == 1 ? "unique" : "tied",
                    "alphas" => [rows[nd[i]]["alpha"] for i in chosen], "score" => best)
            end
        end
    else
        report["log_loss_knee"]["status"] = "nonpositive_loss"
    end
    re = Float64[rows[i]["energy_rmse_meV_per_atom"] for i in nd]
    rf = Float64[rows[i]["force_rmse_meV_per_A"] for i in nd]
    all(isfinite, re) && all(isfinite, rf) && all(>=(0), re) && all(>=(0), rf) ||
        throw(ArgumentError("Physical RMSE values must be finite and nonnegative."))
    xy = normalize_pair(re, rf)
    if isnothing(xy)
        report["linear_rmse_knee"]["status"] = "degenerate_range"
    else
        distances = hypot.(xy[1], xy[2])
        best = minimum(distances)
        chosen = findall(x -> isapprox(x, best; rtol=1e-12, atol=1e-14), distances)
        report["linear_rmse_knee"] = Dict("status" => length(chosen) == 1 ? "unique" : "tied",
            "alphas" => [rows[nd[i]]["alpha"] for i in chosen], "distance" => best)
    end
    return report
end
