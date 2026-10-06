using TOML

function cli_options(args)
    options = Dict{String,Any}("set" => String[])
    flags = ("check", "help", "base-grid")
    values = ("config", "dataset", "solver", "sweep", "input", "output", "set")
    i = 1
    while i <= length(args)
        arg = args[i]
        startswith(arg, "--") || throw(ArgumentError("Expected an option, got $arg"))
        key = arg[3:end]
        if key in flags
            options[key] = true
        elseif key in values
            i < length(args) || throw(ArgumentError("Missing value for --$key"))
            i += 1
            key == "set" ? push!(options["set"], args[i]) : (options[key] = args[i])
        else
            throw(ArgumentError("Unknown option: $arg"))
        end
        i += 1
    end
    return options
end

function cli_overrides(settings)
    updates = Dict{String,Any}()
    for setting in settings
        pair = split(setting, '='; limit=2)
        length(pair) == 2 || throw(ArgumentError("Use --set section.key=value"))
        path, raw = pair
        occursin(r"^[A-Za-z_][A-Za-z_0-9]*(\.[A-Za-z_][A-Za-z_0-9]*)*$", path) ||
            throw(ArgumentError("Invalid configuration key: $path"))
        value = try
            TOML.parse("value = " * raw)["value"]
        catch err
            occursin(r"^[A-Za-z_][A-Za-z_0-9/-]*$", raw) ? raw : rethrow(err)
        end
        keys = split(path, '.')
        current = updates
        for key in keys[1:end-1]
            current = get!(current, key, Dict{String,Any}())
            current isa AbstractDict || throw(ArgumentError("Conflicting overrides at $path"))
        end
        current[last(keys)] = value
    end
    return updates
end

function print_help(kind)
    println("Usage: julia --project=. scripts/$(kind).jl [options]")
    println("  --config PATH       Repository-relative experiment TOML")
    println("  --dataset NAME      licl, water, or si")
    println("  --set KEY=VALUE     Explicit TOML override; repeat as needed")
    println("  --check             Validate configuration without loading data or training")
    kind == "run_scan" && println("  --solver lbfgs|lsqr  --base-grid (omit knee refinement explicitly)")
    kind == "run_ablation" && println("  --sweep force_batch|primal_lr|dual_lr|all")
    kind == "analyze_pareto" && println("  --input PATH        scan.json\n  --output PATH       Analysis JSON destination")
    kind == "prepare_data" && println("Loads data and persists the Julia water split; never trains a model.")
end

function main(kind, args=ARGS)
    try
        options = cli_options(args)
        if get(options, "help", false)
            print_help(kind)
            return 0
        end
        return execute_cli(kind, options)
    catch err
        showerror(stderr, err)
        println(stderr)
        return 1
    end
end

# Kept in a separate include so --help needs only the Julia standard library.
function execute_cli(kind, options)
    include(joinpath(@__DIR__, "..", "src", "ACEUtils.jl"))
    # Functions were defined by include in this call; enter their newest world.
    return Base.invokelatest(_execute_loaded_cli, kind, options)
end

function _execute_loaded_cli(kind, options)
    check = get(options, "check", false)
    if kind == "analyze_pareto"
        haskey(options, "input") && haskey(options, "output") ||
            throw(ArgumentError("Pareto analysis requires --input and --output."))
        payload = ACEUtils.JSON.parsefile(ACEUtils.repo_path(options["input"]))
        rows = payload isa AbstractVector ? payload : payload["runs"]
        report = ACEUtils.analyze_pareto(rows)
        if !check
            ACEUtils._write_json(ACEUtils.repo_path(options["output"]), report)
        end
        ACEUtils.JSON.print(stdout, report, 2)
        println()
        return 0
    end
    defaults = Dict("train" => "pd_licl", "run_scan" => "static", "run_ablation" => "ablation",
                    "run_warmstart" => "warmstart", "prepare_data" => "static")
    haskey(defaults, kind) || throw(ArgumentError("Unknown command: $kind"))
    path = get(options, "config", "configs/experiments/$(defaults[kind]).toml")
    dataset = get(options, "dataset", kind == "prepare_data" ? "water" : nothing)
    config = ACEUtils.load_config(path; dataset, overrides=cli_overrides(options["set"]))
    report = if kind == "train"
        if check
            ACEUtils.validate_config(config)
            Dict("status" => "configuration_valid", "method" => config["method"])
        else
            result = ACEUtils.train_model(config)
            Dict("output_dir" => result["output_dir"], "train" => result["train"], "test" => result["test"])
        end
    elseif kind == "run_scan"
        ACEUtils.run_scan(config; solver=get(options, "solver", "lbfgs"),
                          base_grid=get(options, "base-grid", false), check)
    elseif kind == "run_ablation"
        ACEUtils.run_ablation(config; sweep=get(options, "sweep", "all"), check)
    elseif kind == "run_warmstart"
        ACEUtils.run_warmstart(config; check)
    else
        if check
            spec = config["dataset"]
            files = haskey(spec, "file") ? [spec["file"]] : [spec["train_file"], spec["test_file"]]
            all(f -> isfile(ACEUtils.repo_path(f)), files) || throw(ArgumentError("A dataset file is missing."))
            Dict("status" => "input_files_present", "training_performed" => false)
        else
            data = ACEUtils.load_datasets(config)
            Dict("training_frames" => length(data.train), "test_frames" => length(data.test),
                 "training_performed" => false, "split_file" => get(config["dataset"], "split_file", nothing))
        end
    end
    ACEUtils.JSON.print(stdout, report, 2)
    println()
    return 0
end
