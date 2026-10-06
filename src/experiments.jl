function _require_experiment(config, paths)
    missing = [p for p in paths if is_required(config_value(config, p, "REQUIRED"))]
    isempty(missing) || throw(ArgumentError("Experiment parameters must be supplied: " * join(missing, ", ")))
end

function _prepare_experiment(config)
    Random.seed!(Int(config["seed"]))
    threads = BLAS.get_num_threads()
    BLAS.set_num_threads(Int(config_value(config, "runtime.blas_threads", 1)))
    try
        return prepare_problem(config)
    finally
        BLAS.set_num_threads(threads)
    end
end

function _write_json(path, value)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, value, 2)
    end
    return path
end

function _scan_row(result)
    Dict("alpha" => result["alpha"],
         "train_L_E" => result["train"]["L_E"], "train_L_F" => result["train"]["L_F"],
         "test_L_E" => result["test"]["L_E"], "test_L_F" => result["test"]["L_F"],
         "energy_rmse_meV_per_atom" => result["test"]["energy_rmse_meV_per_atom"],
         "force_rmse_meV_per_A" => result["test"]["force_rmse_meV_per_A"],
         "optimization_seconds" => result["timings"]["optimization_seconds"],
         "output_dir" => result["output_dir"])
end

function scan_alphas(config; base_grid=false)
    _require_experiment(config, ["scan.n", "scan.alpha_min", "scan.alpha_max"])
    spec = config["scan"]
    n, lo, hi = Int(spec["n"]), Float64(spec["alpha_min"]), Float64(spec["alpha_max"])
    n >= 2 && 0 < lo < hi < 1 || throw(ArgumentError("Require scan.n>=2 and 0<alpha_min<alpha_max<1."))
    logit(p) = log(p / (1-p))
    alphas = [1 / (1 + exp(-x)) for x in range(logit(hi), logit(lo); length=n)]
    if !base_grid
        _require_experiment(config, ["scan.knee_alphas"])
        extra = spec["knee_alphas"]
        extra isa AbstractVector && all(a -> a isa Real && isfinite(a) && 0<a<1, extra) ||
            throw(ArgumentError("scan.knee_alphas must be an explicit list of weights in (0,1)."))
        append!(alphas, Float64.(extra))
    end
    return unique(alphas)
end

"""Compute LSQR scan weights with a constant squared row-weight norm."""
function _scan_weights(alpha, problem, config)
    opt = config["scan"]
    e0, f0 = Float64(opt["weight_reference_E"]), Float64(opt["weight_reference_F"])
    e0 > 0 && f0 > 0 || throw(ArgumentError("Scan reference weights must be positive."))
    ne, nf = length(problem.train_info.E_inds), length(problem.train_info.F_inds)
    ke, kf = sum(abs2, problem.train_info.W_E), nf
    constant = e0^2 * ke + f0^2 * kf
    ratio_sq = alpha * nf / ((1-alpha) * ne)
    wf = sqrt(constant / (ratio_sq * ke + kf))
    return (sqrt(ratio_sq) * wf, wf)
end

"""Run independent fixed-alpha fits, including any configured knee weights."""
function run_scan(config; solver="lbfgs", base_grid=false, check=false)
    solver in ("lbfgs", "lsqr") || throw(ArgumentError("Use scan solver lbfgs or lsqr."))
    alphas = scan_alphas(config; base_grid)
    configurations = Dict{String,Any}[]
    if solver == "lsqr"
        _require_experiment(config, ["scan.weight_reference_E", "scan.weight_reference_F"])
        for key in ("weight_reference_E", "weight_reference_F")
            value = config["scan"][key]
            value isa Real && !(value isa Bool) && isfinite(value) && value > 0 ||
                throw(ArgumentError("scan.$key must be finite and positive."))
        end
    end
    for alpha in alphas
        changes = Dict{String,Any}("method" => solver == "lbfgs" ? "static" : "lsqr",
                                  "initialization" => Dict("mode" => "zeros"))
        changes["static"] = Dict("alpha" => alpha)
        # Actual LSQR weights are derived from training observation counts after assembly.
        solver == "lsqr" && (changes["lsqr"] = Dict("alpha" => alpha, "weight_E" => 1.0, "weight_F" => 1.0))
        cfg = merge_config(config, changes)
        validate_config(cfg)
        push!(configurations, cfg)
    end
    check && return Dict("status" => "configuration_valid", "fits" => length(alphas), "alphas" => alphas)
    p = _prepare_experiment(first(configurations))
    destination = _new_output_dir(config, "scan_$(solver)")
    report = Dict{String,Any}("complete" => false, "solver" => solver,
        "base_grid_only" => base_grid, "assembly_seconds" => p.assembly_seconds,
        "runs" => Dict{String,Any}[])
    for (cfg, alpha) in zip(configurations, alphas)
        cfg["output_dir"] = destination
        if solver == "lsqr"
            we, wf = _scan_weights(alpha, p, cfg)
            cfg["lsqr"]["weight_E"], cfg["lsqr"]["weight_F"] = we, wf
        end
        result = train_model(cfg; problem=p)
        push!(report["runs"], _scan_row(result))
        _write_json(joinpath(destination, "scan.json"), report)
    end
    report["pareto"] = analyze_pareto(report["runs"])
    report["optimization_seconds"] = sum(r["optimization_seconds"] for r in report["runs"])
    report["complete"] = true
    _write_json(joinpath(destination, "scan.json"), report)
    return merge(report, Dict("output_dir" => destination))
end

function ablation_cases(config; sweep="all")
    config["dataset"]["name"] == "LiCl" || throw(ArgumentError("This ablation entry point supports LiCl."))
    sweep in ("force_batch", "primal_lr", "dual_lr", "all") || throw(ArgumentError("Unknown ablation sweep: $sweep"))
    cases = Pair{String,Dict{String,Any}}[]
    opt = config["ablation"]
    if sweep in ("force_batch", "all")
        _require_experiment(config, ["ablation.force_batches", "ablation.batch_scaling",
            "ablation.reference_batch_size_F", "ablation.reference_primal_lr", "ablation.reference_dual_lr"])
        scaling = opt["batch_scaling"]
        scaling in ("primal", "dual", "both") || throw(ArgumentError("batch_scaling must specify primal, dual, or both for linear scaling."))
        anchor = Float64(opt["reference_batch_size_F"])
        anchor > 0 || throw(ArgumentError("Reference batch size must be positive."))
        for batch in opt["force_batches"]
            batch isa Integer && batch > 0 || throw(ArgumentError("Force batches must be positive integers."))
            factor = batch / anchor
            primal = Float64(opt["reference_primal_lr"]) * (scaling in ("primal", "both") ? factor : 1)
            dual = Float64(opt["reference_dual_lr"]) * (scaling in ("dual", "both") ? factor : 1)
            push!(cases, "force_batch_$(batch)" => Dict("pd" => Dict("batch_size_F" => batch,
                "primal_lr" => primal, "initial_learning_rate" => primal, "dual_lr" => dual)))
        end
    end
    if sweep in ("primal_lr", "all")
        for rate in opt["primal_lrs"]
            push!(cases, "primal_lr_$(rate)" => Dict("pd" => Dict("primal_lr" => rate, "initial_learning_rate" => rate)))
        end
    end
    if sweep in ("dual_lr", "all")
        for rate in opt["dual_lrs"]
            push!(cases, "dual_lr_$(rate)" => Dict("pd" => Dict("dual_lr" => rate)))
        end
    end
    isempty(cases) && throw(ArgumentError("The ablation contains no cases."))
    return [label => merge_config(config, merge_config(changes,
        Dict("method" => "pd", "initialization" => Dict("mode" => "zeros")))) for (label,changes) in cases]
end

function _target_time(pd_result; threshold_meV_per_A=200.0)
    isnothing(pd_result) && return nothing
    history = pd_result["history"]
    for i in eachindex(history["test_L_F"])
        if history["test_evaluated"][i] && sqrt(history["test_L_F"][i]) * 1000 <= threshold_meV_per_A
            return history["elapsed_seconds"][i]
        end
    end
    return nothing
end

function run_ablation(config; sweep="all", check=false)
    cases = ablation_cases(config; sweep)
    for (_, cfg) in cases
        validate_config(cfg)
    end
    check && return Dict("status" => "configuration_valid", "cases" => first.(cases))
    p = _prepare_experiment(last(first(cases)))
    destination = _new_output_dir(config, "ablation_$(sweep)")
    report = Dict{String,Any}("complete" => false, "sweep" => sweep,
        "assembly_seconds" => p.assembly_seconds, "runs" => Dict{String,Any}[])
    for (label, cfg) in cases
        cfg["output_dir"] = joinpath(destination, label)
        result = train_model(cfg; problem=p)
        push!(report["runs"], Dict("case" => label, "output_dir" => result["output_dir"],
            "test" => result["test"], "timings" => result["timings"],
            "first_evaluated_force_rmse_200_time_seconds" => _target_time(result["pd"])))
        _write_json(joinpath(destination, "ablation.json"), report)
    end
    report["complete"] = true
    _write_json(joinpath(destination, "ablation.json"), report)
    return merge(report, Dict("output_dir" => destination))
end

function warmstart_cases(config)
    config["dataset"]["name"] == "LiCl" || throw(ArgumentError("This warm-start entry point supports LiCl."))
    _require_experiment(config, ["warmstart.labels", "warmstart.alphas", "warmstart.max_iter", "warmstart.gamma", "warmstart.f_reltol"])
    spec = config["warmstart"]
    length(spec["labels"]) == length(spec["alphas"]) == 3 || throw(ArgumentError("Provide exactly three warm-start labels and alphas."))
    all(label -> label isa AbstractString && occursin(r"^[A-Za-z0-9_-]+$", label), spec["labels"]) ||
        throw(ArgumentError("Warm-start labels must be simple directory-safe names."))
    length(unique(spec["labels"])) == 3 || throw(ArgumentError("Warm-start labels must be distinct."))
    cases = []
    for (label, alpha) in zip(spec["labels"], spec["alphas"])
        source = merge_config(config, Dict("method" => "static", "initialization" => Dict("mode" => "zeros"),
            "static" => Dict("alpha" => alpha, "max_iter" => spec["max_iter"],
                "gamma" => spec["gamma"], "f_reltol" => spec["f_reltol"])))
        target = merge_config(config, Dict("method" => "pd", "initialization" => Dict("mode" => "warmstart")))
        validate_config(source)
        validate_config(target)
        push!(cases, (; label, source, target))
    end
    return cases
end

function run_warmstart(config; check=false)
    cases = warmstart_cases(config)
    check && return Dict("status" => "configuration_valid", "cases" => [x.label for x in cases])
    p = _prepare_experiment(first(cases).source)
    destination = _new_output_dir(config, "warmstart")
    report = Dict{String,Any}("complete" => false, "assembly_seconds" => p.assembly_seconds, "runs" => Any[])
    for case in cases
        case.source["output_dir"] = joinpath(destination, case.label, "source")
        initial = train_model(case.source; problem=p)
        case.target["output_dir"] = joinpath(destination, case.label, "pd")
        case.target["initialization"]["source_alpha"] = initial["alpha"]
        case.target["initialization"]["source_output_dir"] = initial["output_dir"]
        result = train_model(case.target; theta_init=copy(initial["coeffs"]), problem=p)
        push!(report["runs"], Dict("case" => case.label, "source_output_dir" => initial["output_dir"],
            "output_dir" => result["output_dir"], "test" => result["test"], "timings" => result["timings"]))
        _write_json(joinpath(destination, "warmstart.json"), report)
    end
    report["complete"] = true
    _write_json(joinpath(destination, "warmstart.json"), report)
    return merge(report, Dict("output_dir" => destination))
end
