file_sha256(path::AbstractString) = open(path, "r") do io
    bytes2hex(SHA.sha256(io))
end

function validate_split_indices(split::AbstractDict, n_total::Integer, n_train::Integer, n_test::Integer)
    train = get(split, "train_indices", Int[])
    test = get(split, "test_indices", Int[])
    length(train) == n_train || throw(ArgumentError("Split has an incorrect training count"))
    length(test) == n_test || throw(ArgumentError("Split has an incorrect test count"))
    all(i -> i isa Integer && !(i isa Bool) && 1 <= i <= n_total, vcat(train, test)) ||
        throw(ArgumentError("Split indices must be 1-based integers within the dataset"))
    length(unique(train)) == n_train || throw(ArgumentError("Training split contains repeated indices"))
    length(unique(test)) == n_test || throw(ArgumentError("Test split contains repeated indices"))
    isempty(intersect(train, test)) || throw(ArgumentError("Training and test splits overlap"))
    sort(vcat(train, test)) == collect(1:n_total) || throw(ArgumentError("Split does not cover the dataset exactly"))
    return (train_indices=Int.(train), test_indices=Int.(test))
end

"""Read or generate a persisted dataset partition using Julia's randperm.

Generation is isolated in a Task so Random.seed! does not reseed the caller's
task-local RNG. Persisted indices are authoritative across Julia versions.
"""
function load_or_create_split(config::AbstractDict, n_total::Integer)
    spec = config["dataset"]
    source = repo_path(spec["file"])
    split_file = repo_path(spec["split_file"])
    sha = file_sha256(source)
    n_train, n_test = spec["expected_train"], spec["expected_test"]
    n_total == n_train + n_test || throw(ArgumentError("Unexpected dataset length for the requested split"))
    seed = Int(spec["split_seed"])
    if isfile(split_file)
        split = TOML.parsefile(split_file)
        get(split, "source_sha256", "") == sha || throw(ArgumentError("Saved split source SHA-256 differs from $source; do not silently regenerate it"))
        get(split, "seed", nothing) == seed || throw(ArgumentError("Saved split seed differs from dataset.split_seed"))
        get(split, "algorithm", "") == "Random.seed! + randperm" || throw(ArgumentError("Unrecognized saved split algorithm"))
        get(split, "n_total", nothing) == n_total || throw(ArgumentError("Saved split dataset length differs"))
    else
        task = Task() do
            Random.seed!(seed)
            randperm(n_total)
        end
        schedule(task)
        indices = fetch(task)
        split = Dict{String,Any}(
            "source_file" => spec["file"], "source_sha256" => sha,
            "julia_version" => string(VERSION), "seed" => seed,
            "algorithm" => "Random.seed! + randperm", "n_total" => n_total,
            "train_indices" => indices[1:n_train],
            "test_indices" => indices[n_train+1:end],
            "provenance" => "Generated with Julia from dataset.split_seed using Random.seed! + randperm",
        )
        validate_split_indices(split, n_total, n_train, n_test)
        mkpath(dirname(split_file))
        temporary, io = mktemp(dirname(split_file))
        try
            TOML.print(io, split; sorted=true)
            close(io)
            mv(temporary, split_file; force=false)
        finally
            isopen(io) && close(io)
            isfile(temporary) && rm(temporary)
        end
    end
    indices = validate_split_indices(split, n_total, n_train, n_test)
    return merge(indices, (provenance=split,))
end

function label_options(spec::AbstractDict)
    return (energy_key=String(spec["energy_key"]), force_key=String(spec["force_key"]), virial_key=nothing)
end

"""Load raw systems and split-specific label names without filtering configurations."""
function load_datasets(config::AbstractDict)
    spec = config["dataset"]
    provenance = Dict{String,Any}("dataset" => spec["name"], "units" => deepcopy(config["units"]))
    if haskey(spec, "file")
        path = repo_path(spec["file"])
        isfile(path) || throw(ArgumentError("Dataset does not exist: $path"))
        dataset = ExtXYZ.load(path)
        split = load_or_create_split(config, length(dataset))
        train, test = dataset[split.train_indices], dataset[split.test_indices]
        provenance["split"] = split.provenance
    else
        train_path, test_path = repo_path(spec["train_file"]), repo_path(spec["test_file"])
        for path in (train_path, test_path)
            isfile(path) || throw(ArgumentError("Dataset does not exist: $path"))
        end
        train, test = ExtXYZ.load(train_path), ExtXYZ.load(test_path)
        provenance["train_file"] = spec["train_file"]
        provenance["test_file"] = spec["test_file"]
        provenance["train_sha256"] = file_sha256(train_path)
        provenance["test_sha256"] = file_sha256(test_path)
    end
    length(train) == spec["expected_train"] || throw(ArgumentError("Unexpected training count: $(length(train)); expected $(spec["expected_train"])"))
    length(test) == spec["expected_test"] || throw(ArgumentError("Unexpected test count: $(length(test)); expected $(spec["expected_test"])"))
    isempty(train) && throw(ArgumentError("Training set is empty"))
    isempty(test) && throw(ArgumentError("Test set is empty"))
    return (train=train, test=test,
            train_opts=label_options(spec["train_opts"]),
            test_opts=label_options(spec["test_opts"]), provenance=provenance)
end

"""Construct every AtomsData record and require energy plus all force components.

ACEpotentials 0.9.1 resolves case-insensitive keys per frame. Its constructor
otherwise silently changes missing keys to nothing, so validate both beforehand
and afterwards. Unit weights keep assembled A and y separate from optimization
loss normalization and LSQR weights.
"""
function make_labeled_data(dataset, model, opts)
    isempty(dataset) && throw(ArgumentError("Cannot assemble an empty dataset"))
    for (i, system) in enumerate(dataset)
        length(system) > 0 || throw(ArgumentError("Frame $i has no atoms"))
        available = unique(vcat(collect(keys(system)), collect(AtomsBase.atomkeys(system))))
        for wanted in (opts.energy_key, opts.force_key)
            matches = filter(key -> lowercase(string(key)) == lowercase(string(wanted)), available)
            length(matches) == 1 || throw(ArgumentError("Frame $i: expected exactly one case-insensitive '$wanted' label, found $(length(matches))"))
        end
    end
    weights = Dict("default" => Dict("E" => 1.0, "F" => 1.0, "V" => 1.0))
    data = ACEpotentials.make_atoms_data(dataset, model; opts..., weights=weights)
    length(data) == length(dataset) || throw(ArgumentError("Label conversion changed the number of frames"))
    for (i, d) in enumerate(data)
        isnothing(d.energy_key) && throw(ArgumentError("Frame $i is missing energy label $(opts.energy_key)"))
        isnothing(d.force_key) && throw(ArgumentError("Frame $i is missing force label $(opts.force_key)"))
        isnothing(d.virial_key) || throw(ArgumentError("Virial observations are not part of this experiment"))
        target = ACEfit.target_vector(d)
        length(target) == 1 + 3length(d.system) || throw(ArgumentError("Frame $i has an invalid observation count"))
        all(isfinite, target) || throw(ArgumentError("Frame $i contains non-finite labels"))
    end
    return data
end

"""Return E/F row indices into ACEfit's raw observation-by-feature matrix.

Each frame's row range is assigned in input order, with energy, force, then
virial rows inside a frame. Index vectors traverse frame packets in descending
size order. Sorting packets does not renumber rows: it only reorders the returned
index vectors.
"""
function get_data_indices(data::AbstractVector{<:ACEpotentials.AtomsData})
    isempty(data) && throw(ArgumentError("Cannot index an empty dataset"))
    starts = Int[]
    nextrow = 1
    for d in data
        push!(starts, nextrow)
        nextrow += ACEfit.count_observations(d)
    end
    order = sortperm(eachindex(data); by=i -> length(data[i]), rev=true)
    E_inds, F_inds, N_atoms_E = Int[], Int[], Int[]
    W_E = Float64[]
    for i in order
        d, row = data[i], starts[i]
        n = length(d.system)
        n > 0 || throw(ArgumentError("Cannot index a zero-atom frame"))
        if !isnothing(d.energy_key)
            push!(E_inds, row)
            push!(W_E, 1 / sqrt(n))
            push!(N_atoms_E, n)
            row += 1
        end
        if !isnothing(d.force_key)
            append!(F_inds, row:row+3n-1)
            row += 3n
        end
        !isnothing(d.virial_key) && (row += 6)
        row == starts[i] + ACEfit.count_observations(d) || throw(ArgumentError("Observation layout mismatch"))
    end
    isempty(E_inds) && throw(ArgumentError("Dataset has no energy observations"))
    isempty(F_inds) && throw(ArgumentError("Dataset has no force observations"))
    return (E_inds=E_inds, F_inds=F_inds, W_E=W_E, N_atoms_E=N_atoms_E, row_count=nextrow-1)
end
