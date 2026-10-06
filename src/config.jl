"""Recursively merge string-keyed dictionaries without modifying either input."""
function merge_config(base::AbstractDict, updates::AbstractDict)
    result = Dict{String,Any}(string(k) => deepcopy(v) for (k, v) in base)
    for (key, value) in updates
        k = string(key)
        result[k] = value isa AbstractDict && get(result, k, nothing) isa AbstractDict ?
            merge_config(result[k], value) : deepcopy(value)
    end
    return result
end

repo_path(path::AbstractString) = isabspath(path) ? normpath(path) : normpath(joinpath(REPO_ROOT, path))

"""Load a dataset configuration, overlay the experiment, then explicit overrides.

Paths in TOML are repository-relative. `dataset` selects licl, water, or si and
overrides the experiment's dataset_config. TOML batch value "all" means nothing.
This function only reads configuration; it does not start training or load data.
"""
function load_config(path::AbstractString; dataset=nothing, overrides=Dict())
    config_file = repo_path(path)
    isfile(config_file) || throw(ArgumentError("Configuration file does not exist: $config_file"))
    experiment = TOML.parsefile(config_file)
    dataset_file = if isnothing(dataset)
        get(experiment, "dataset_config", nothing)
    else
        name = lowercase(string(dataset))
        name in ("licl", "water", "si") || throw(ArgumentError("Unknown dataset: $dataset"))
        "configs/datasets/$name.toml"
    end
    base = isnothing(dataset_file) ? Dict{String,Any}() : TOML.parsefile(repo_path(dataset_file))
    config = merge_config(merge_config(base, experiment), overrides)
    !isnothing(dataset_file) && (config["dataset_config"] = dataset_file)
    config["config_file"] = relpath(config_file, REPO_ROOT)
    if haskey(config, "pd")
        for key in ("batch_size_E", "batch_size_F")
            get(config["pd"], key, nothing) == "all" && (config["pd"][key] = nothing)
        end
    end
    return config
end

function config_value(config::AbstractDict, path::AbstractString, default=nothing)
    value = config
    for key in split(path, '.')
        value isa AbstractDict && haskey(value, key) || return default
        value = value[key]
    end
    return value
end

is_required(value) = value isa AbstractString && startswith(value, "REQUIRED")

"""Validate the selected experiment before matrix assembly; report all missing fields."""
function validate_config(config::AbstractDict; method=get(config, "method", "pd"))
    errors = String[]
    function need(path; predicate=x -> true, description="has an invalid value")
        value = config_value(config, path, missing)
        if ismissing(value) || is_required(value)
            push!(errors, "$path: required; set this field in the configuration or with --set")
        elseif !predicate(value)
            push!(errors, "$path: $description")
        end
    end
    function optional(path; predicate=x -> true, description="has an invalid value")
        ismissing(config_value(config, path, missing)) || need(path; predicate, description)
    end
    positive = x -> x isa Real && !(x isa Bool) && isfinite(x) && x > 0
    nonnegative = x -> x isa Real && !(x isa Bool) && isfinite(x) && x >= 0
    positive_int = x -> x isa Integer && !(x isa Bool) && x > 0
    nonnegative_int = x -> x isa Integer && !(x isa Bool) && x >= 0
    boolean = x -> x isa Bool
    function lbfgs_options(prefix)
        need("$prefix.max_iter"; predicate=positive_int)
        need("$prefix.gamma"; predicate=nonnegative)
        need("$prefix.f_reltol"; predicate=nonnegative)
        optional("$prefix.g_tol"; predicate=nonnegative)
        optional("$prefix.verbose"; predicate=boolean)
        for key in ("eval_every", "print_every")
            optional("$prefix.$key"; predicate=positive_int)
        end
    end
    need("dataset.name")
    need("model.elements"; predicate=x -> x isa AbstractVector && !isempty(x))
    need("model.order"; predicate=positive_int)
    need("model.totaldegree"; predicate=positive_int)
    need("model.rcut"; predicate=positive)
    for part in ("train", "test"), label in ("energy_key", "force_key")
        need("dataset.$(part)_opts.$label"; predicate=x -> x isa AbstractString && !isempty(x))
    end
    for part in ("train", "test")
        need("dataset.expected_$part"; predicate=positive_int)
    end
    config_value(config, "units.confirmed", false) === true ||
        push!(errors, "units.confirmed: input energy, force, and coordinate units must be verified before training")
    for (key, expected) in (("energy", "eV"), ("force", "eV/angstrom"), ("length", "angstrom"))
        need("units.$key"; predicate=x -> x == expected,
             description="input must be documented in $expected; convert source data explicitly before training")
    end
    need("seed"; predicate=nonnegative_int)
    mode = config_value(config, "initialization.mode", "zeros")
    mode in ("zeros", "warmstart") || push!(errors, "initialization.mode: use zeros or warmstart")
    need("runtime.blas_threads"; predicate=positive_int)
    need("output_dir"; predicate=x -> x isa AbstractString && !isempty(x))

    if method == "pd"
        for key in ("primal_lr", "dual_lr", "initial_learning_rate", "lr_decay_factor")
            need("pd.$key"; predicate=positive)
        end
        for key in ("max_iter", "dual_update_freq", "eval_every")
            need("pd.$key"; predicate=positive_int)
        end
        need("pd.lr_decay_step"; predicate=nonnegative_int)
        need("pd.warmup_steps"; predicate=nonnegative_int)
        optional("pd.tol"; predicate=nonnegative)
        optional("pd.patience"; predicate=positive_int)
        optional("pd.verbose"; predicate=boolean)
        for key in ("epsilon_F", "lambda_F_0")
            need("pd.$key"; predicate=nonnegative)
        end
        need("pd.lambda_F_max"; predicate=positive)
        initial_lambda = config_value(config, "pd.lambda_F_0")
        maximum_lambda = config_value(config, "pd.lambda_F_max")
        if nonnegative(initial_lambda) && positive(maximum_lambda) && initial_lambda > maximum_lambda
            push!(errors, "pd.lambda_F_0: must not exceed pd.lambda_F_max")
        end
        for key in ("batch_size_E", "batch_size_F")
            need("pd.$key"; predicate=x -> isnothing(x) || x == "all" || positive_int(x),
                 description="use a positive observation count, or TOML \"all\" for a full batch")
        end
        optional("refine.enabled"; predicate=boolean)
        if config_value(config, "refine.enabled", true) !== false
            lbfgs_options("refine")
        end
    elseif method == "static"
        need("static.alpha"; predicate=x -> x isa Real && 0 < x < 1)
        lbfgs_options("static")
    elseif method == "lsqr"
        for key in ("weight_E", "weight_F", "atol")
            need("lsqr.$key"; predicate=positive)
        end
        need("lsqr.damp"; predicate=nonnegative)
        need("lsqr.smoothness"; predicate=nonnegative)
        need("lsqr.conlim"; predicate=positive)
        need("lsqr.maxiter"; predicate=positive_int)
        need("lsqr.verbose"; predicate=boolean)
    elseif method == "heuristic"
        need("heuristic.alpha"; predicate=x -> x isa Real && 0 < x < 1)
        lbfgs_options("static")
    else
        push!(errors, "method: unsupported method '$method' (use pd, static, lsqr, or heuristic)")
    end
    isempty(errors) || throw(ArgumentError("Configuration is incomplete or invalid:\n  - " * join(errors, "\n  - ")))
    return config
end
