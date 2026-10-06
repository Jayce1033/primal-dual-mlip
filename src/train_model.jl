_problem_signature(config) = Dict(key => deepcopy(config[key]) for key in ("dataset", "model", "units"))

"""Assemble raw, reference-energy-adjusted matrices once for a fixed basis."""
function prepare_problem(config)
    started = time_ns()
    data = load_datasets(config)
    model = build_model(config)
    train_data = make_labeled_data(data.train, model, data.train_opts)
    test_data = make_labeled_data(data.test, model, data.test_opts)
    train_info, test_info = get_data_indices(train_data), get_data_indices(test_data)
    if get(config, "method", "pd") == "pd"
        for (key, count) in (("batch_size_E", length(train_info.E_inds)),
                             ("batch_size_F", length(train_info.F_inds)))
            batch = _batch_size(config["pd"][key])
            isnothing(batch) || batch <= count ||
                throw(ArgumentError("pd.$key exceeds the available $count observations."))
        end
    end
    A_train, y_train, _ = ACEfit.assemble(train_data, model)
    A_test, y_test, _ = ACEfit.assemble(test_data, model)
    validate_observations(A_train, y_train, train_info)
    validate_observations(A_test, y_test, test_info)
    return (; model, A_train, y_train, train_info, A_test, y_test, test_info,
        provenance=data.provenance, input_signature=_problem_signature(config),
        assembly_seconds=(time_ns() - started) / 1e9)
end

_batch_size(value) = isnothing(value) || value == "all" ? nothing : Int(value)

function _new_output_dir(config, label)
    parent = repo_path(get(config, "output_dir", "results"))
    stamp = Dates.format(now(UTC), "yyyymmddTHHMMSSsss")
    destination = joinpath(parent, "$(label)_$(stamp)_$(first(string(uuid4()), 8))")
    mkpath(destination)
    return destination
end

function _optimizer_status(result)
    Dict("converged" => Optim.converged(result), "iterations" => Optim.iterations(result),
         "final_objective" => Optim.minimum(result))
end

function _lbfgs_options(section)
    (; max_iter=Int(section["max_iter"]), g_tol=Float64(get(section, "g_tol", 1e-4)),
       f_reltol=Float64(get(section, "f_reltol", 0.0)),
       l2_reg=Float64(get(section, "gamma", 1e-8)),
       verbose=Bool(get(section, "verbose", true)),
       eval_every=Int(get(section, "eval_every", 10)),
       print_every=Int(get(section, "print_every", 50)))
end

function _toml_safe(value)
    isnothing(value) && return "all"
    value isa AbstractDict && return Dict(string(k) => _toml_safe(v) for (k,v) in value)
    value isa AbstractVector && return [_toml_safe(v) for v in value]
    return value
end

function _save_training_result!(result, config)
    haskey(result["model"].model.meta, "model_spec") ||
        error("The fitted model is missing construction metadata required by load_model.")
    destination = _new_output_dir(config, "$(config["dataset"]["name"])_$(config["method"])")
    result["output_dir"] = destination
    open(joinpath(destination, "resolved_config.toml"), "w") do io
        TOML.print(io, _toml_safe(config); sorted=true)
    end
    open(joinpath(destination, "summary.json"), "w") do io
        JSON.print(io, Dict(k => result[k] for k in
            ("method", "lambda", "alpha", "train", "test", "timings", "status", "provenance")), 2)
    end
    # BSON preserves optimizer state and the final PD gradient's explicit NaN.
    histories = Dict("coeffs" => result["coeffs"], "lambda" => result["lambda"],
        "alpha" => result["alpha"], "pd" => result["pd"], "refinement" => result["refinement"])
    BSON.bson(joinpath(destination, "training.bson"), histories)
    ACEpotentials.save_model(result["model"], joinpath(destination, "model.json");
        meta=Dict("method" => config["method"], "units" => config["units"]),
        verbose=false, save_project=false)
    return result
end

"""Fit from explicit configuration; importing this file never starts a fit.

A supplied `problem` reuses fixed-basis assembly. Each call starts fresh
optimizer states. Initial coefficients are explicit for a warm start.
"""
function train_model(config; overrides=Dict{String,Any}(), theta_init=nothing, problem=nothing)
    config = merge_config(config, overrides)
    validate_config(config)
    mode = config_value(config, "initialization.mode", "zeros")
    mode == "warmstart" && isnothing(theta_init) &&
        throw(ArgumentError("A warm start requires coefficients from its configured source fit."))
    mode == "zeros" && !isnothing(theta_init) &&
        throw(ArgumentError("Set initialization.mode=warmstart when supplying theta_init."))
    Random.seed!(Int(config["seed"]))
    old_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(Int(config_value(config, "runtime.blas_threads", 1)))
    try
        reused = !isnothing(problem)
        if reused
            hasproperty(problem, :input_signature) && problem.input_signature == _problem_signature(config) ||
                throw(ArgumentError("The cached problem does not match the requested dataset, basis, or units. Reassemble it."))
        end
        p = reused ? problem : prepare_problem(config)
        theta0 = isnothing(theta_init) ? zeros(size(p.A_train, 2)) : Float64.(theta_init)
        length(theta0) == size(p.A_train, 2) || throw(DimensionMismatch("Initial coefficient count does not match basis."))
        all(isfinite, theta0) || throw(ArgumentError("Initial coefficients must be finite."))
        args = (p.A_train, p.y_train, p.train_info, p.A_test, p.y_test, p.test_info)
        pd_result, refinement, lambda, alpha = nothing, nothing, nothing, nothing
        status = Dict{String,Any}()
        timings = Dict("assembly_seconds" => reused ? 0.0 : p.assembly_seconds,
                       "pd_seconds" => 0.0, "refinement_seconds" => 0.0,
                       "linear_solve_seconds" => 0.0, "layout_seconds" => 0.0)
        method = config["method"]
        if method == "pd"
            opt = config["pd"]
            solver = ConstrainedPDSolver(;
                primal_opt_rule=Optimisers.Adam(opt["primal_lr"]),
                dual_opt_rule=Optimisers.Adam(opt["dual_lr"]),
                max_iter=Int(opt["max_iter"]), epsilon_F=opt["epsilon_F"],
                lambda_F_0=opt["lambda_F_0"], lambda_F_max=opt["lambda_F_max"],
                dual_update_freq=Int(opt["dual_update_freq"]),
                initial_learning_rate=opt["initial_learning_rate"],
                lr_decay_step=Int(opt["lr_decay_step"]), lr_decay_factor=opt["lr_decay_factor"],
                tol=Float64(get(opt, "tol", 1e-6)), patience=Int(get(opt, "patience", 50)),
                verbose=Bool(get(opt, "verbose", true)))
            started = time_ns()
            A_train_t, A_test_t = permutedims(p.A_train), permutedims(p.A_test)
            timings["layout_seconds"] = (time_ns() - started) / 1e9
            # Separate batch randomness from model construction and matrix reuse.
            Random.seed!(Int(config["seed"]))
            started = time_ns()
            pd_result = solve_pd2_hpc_complete1(solver,
                A_train_t, p.y_train, p.train_info, A_test_t, p.y_test, p.test_info;
                theta_init=theta0, batch_size_E=_batch_size(opt["batch_size_E"]),
                batch_size_F=_batch_size(opt["batch_size_F"]),
                eval_every=Int(opt["eval_every"]), warmup_steps=Int(opt["warmup_steps"]))
            timings["pd_seconds"] = (time_ns() - started) / 1e9
            theta, lambda = pd_result["C"], pd_result["lambda_F"]
            status["pd"] = Dict("iterations" => pd_result["iterations"], "reason" => pd_result["stopping_reason"])
            if get(config["refine"], "enabled", true)
                started = time_ns()
                refinement = solve_only_lbfgs(copy(theta), Float64(lambda), args...;
                    _lbfgs_options(config["refine"])...)
                timings["refinement_seconds"] = (time_ns() - started) / 1e9
                theta = refinement["theta"]
                status["refinement"] = _optimizer_status(refinement["optim_result"])
            end
        elseif method == "static" || method == "heuristic"
            alpha = Float64(method == "static" ? config["static"]["alpha"] : config["heuristic"]["alpha"])
            started = time_ns()
            refinement = solve_static_lbfgs(theta0, alpha, args...; _lbfgs_options(config["static"])...)
            timings["refinement_seconds"] = (time_ns() - started) / 1e9
            theta = refinement["theta"]
            status["static"] = _optimizer_status(refinement["optim_result"])
        elseif method == "lsqr"
            opt = config["lsqr"]
            weights = zeros(length(p.y_train))
            weights[p.train_info.E_inds] .= Float64(opt["weight_E"]) .* p.train_info.W_E
            weights[p.train_info.F_inds] .= Float64(opt["weight_F"])
            prior = ACEpotentials.Models.algebraic_smoothness_prior(p.model.model;
                p=Float64(get(opt, "smoothness", 4)))
            started = time_ns()
            solved = ACEfit.solve(ACEfit.LSQR(damp=Float64(opt["damp"]), atol=Float64(opt["atol"]),
                conlim=Float64(opt["conlim"]), maxiter=Int(opt["maxiter"]), verbose=Bool(opt["verbose"])),
                weights .* (p.A_train / prior), weights .* p.y_train)
            theta = prior \ solved["C"]
            timings["linear_solve_seconds"] = (time_ns() - started) / 1e9
            alpha = get(opt, "alpha", nothing)
            status["lsqr"] = Dict("solver" => "ACEfit.LSQR", "damp" => opt["damp"], "atol" => opt["atol"])
        else
            throw(ArgumentError("Unsupported method: $method"))
        end
        timings["optimization_seconds"] = timings["pd_seconds"] + timings["refinement_seconds"] + timings["linear_solve_seconds"]
        all(isfinite, theta) || error("Solver returned non-finite coefficients.")
        fitted_model = deepcopy(p.model)
        ACEpotentials.Models.set_linear_parameters!(fitted_model, theta)
        started = time_ns()
        train_metrics = evaluate_fit(theta, p.A_train, p.y_train, p.train_info)
        test_metrics = evaluate_fit(theta, p.A_test, p.y_test, p.test_info)
        timings["final_evaluation_seconds"] = (time_ns() - started) / 1e9
        if !isnothing(refinement)
            refinement = Dict(k => v for (k,v) in refinement if k != "optim_result")
        end
        provenance = merge(Dict{String,Any}(p.provenance), Dict("julia_version" => string(VERSION),
            "julia_threads" => Threads.nthreads(), "blas_threads" => BLAS.get_num_threads(),
            "project_sha256" => file_sha256(joinpath(REPO_ROOT, "Project.toml")),
            "manifest_sha256" => file_sha256(joinpath(REPO_ROOT, "Manifest.toml"))))
        result = Dict{String,Any}("method" => method, "model" => fitted_model, "coeffs" => theta,
            "lambda" => lambda, "alpha" => alpha, "pd" => pd_result, "refinement" => refinement,
            "train" => train_metrics, "test" => test_metrics, "status" => status,
            "timings" => timings, "provenance" => provenance)
        return _save_training_result!(result, config)
    finally
        BLAS.set_num_threads(old_blas_threads)
    end
end
