@testset "Dataset-specific configuration and fail-fast validation" begin
    for dataset in ("licl", "water", "si")
        main_config = ACEUtils.load_config("configs/experiments/pd_$(dataset).toml")
        @test ACEUtils.validate_config(main_config) === main_config
    end
    config = ACEUtils.load_config("configs/experiments/pd_licl.toml")
    @test config["pd"]["batch_size_E"] == 117
    @test config["seed"] == 1234
    @test config["pd"]["warmup_steps"] == 1000
    @test config["pd"]["lambda_F_max"] == 1e4
    @test config["refine"]["max_iter"] == 2000
    @test config["refine"]["f_reltol"] == 0.0
    @test config["static"]["f_reltol"] == 1e-6
    @test config["model"]["rcut"] == 5.5
    @test !haskey(config["model"], "Eref")
    incomplete = ACEUtils.merge_config(config, Dict("pd" => Dict("batch_size_F" => "REQUIRED")))
    @test_throws ArgumentError ACEUtils.validate_config(incomplete)

    overrides = Dict("seed" => 42,
        "pd" => Dict("max_iter" => 2, "lr_decay_step" => 0,
                     "lr_decay_factor" => 1.0, "warmup_steps" => 0,
                     "batch_size_F" => "all"),
        "refine" => Dict("max_iter" => 2))
    ready = ACEUtils.load_config("configs/experiments/pd_licl.toml"; overrides=overrides)
    @test isnothing(ready["pd"]["batch_size_F"])
    @test ACEUtils.validate_config(ready) === ready
    @test overrides["pd"]["batch_size_F"] == "all"
    @test_throws ArgumentError ACEUtils.validate_config(
        ACEUtils.merge_config(ready, Dict("pd" => Dict("batch_size_F" => 0))))
    for (section, key, value) in (
        ("pd", "tol", -1.0), ("pd", "patience", 2.5),
        ("pd", "verbose", "REQUIRED"), ("pd", "primal_lr", true),
        ("pd", "lambda_F_max", 0.0), ("pd", "lambda_F_max", Inf),
        ("pd", "lambda_F_max", true), ("pd", "lambda_F_0", 1e4 + 1),
        ("refine", "enabled", "REQUIRED"), ("refine", "verbose", 1),
        ("refine", "g_tol", Inf), ("refine", "eval_every", 0),
        ("refine", "print_every", true))
        @test_throws ArgumentError ACEUtils.validate_config(
            ACEUtils.merge_config(ready, Dict(section => Dict(key => value))))
    end
    @test_throws ArgumentError ACEUtils.validate_config(
        ACEUtils.merge_config(ready, Dict("seed" => true)))

    silicon = ACEUtils.load_config("configs/experiments/static.toml"; dataset="si",
        overrides=Dict("seed" => 42, "static" => Dict("alpha" => 0.5)))
    @test silicon["dataset"]["expected_train"] == 2475
    @test silicon["dataset"]["test_file"] == "data/OOD-Si/Si_test_filtered.xyz"
    @test silicon["dataset"]["train_opts"]["energy_key"] == "dft_energy"
    @test silicon["dataset"]["test_opts"]["energy_key"] == "energy"
    @test silicon["model"]["Eref"]["Si"] == -158.54496821
    @test silicon["static"]["max_iter"] == 10000
    @test ACEUtils.validate_config(silicon) === silicon
    for (key, value) in (("gamma", true), ("g_tol", "REQUIRED"),
                         ("eval_every", 0), ("print_every", -2),
                         ("verbose", "false"), ("max_iter", true))
        @test_throws ArgumentError ACEUtils.validate_config(
            ACEUtils.merge_config(silicon, Dict("static" => Dict(key => value))))
    end
    zero_tolerances = ACEUtils.merge_config(silicon,
        Dict("static" => Dict("g_tol" => 0.0, "f_reltol" => 0.0)))
    @test ACEUtils.validate_config(zero_tolerances) === zero_tolerances

    lsqr = ACEUtils.load_config("configs/experiments/lsqr.toml";
        overrides=Dict("seed" => 42,
            "lsqr" => Dict("weight_E" => 30.0, "weight_F" => 1.0, "smoothness" => 2.5)))
    @test ACEUtils.validate_config(lsqr) === lsqr
    for (key, value) in (("smoothness", -0.5), ("maxiter", 1.5),
                         ("conlim", Inf), ("verbose", 1), ("atol", true))
        @test_throws ArgumentError ACEUtils.validate_config(
            ACEUtils.merge_config(lsqr, Dict("lsqr" => Dict(key => value))))
    end

    water = ACEUtils.load_config("configs/experiments/static.toml"; dataset="water",
        overrides=Dict("seed" => 42, "static" => Dict("alpha" => 0.5)))
    @test water["dataset"]["split_seed"] == 42
    @test water["model"]["order"] == 3
    @test water["model"]["totaldegree"] == 15
    @test water["model"]["rcut"] == 5.5
    @test water["units"]["confirmed"] === true
    @test ACEUtils.validate_config(water) === water
    unconfirmed = ACEUtils.merge_config(water, Dict("units" => Dict("confirmed" => false)))
    @test_throws ArgumentError ACEUtils.validate_config(unconfirmed)
end

@testset "Split index integrity" begin
    good = Dict("train_indices" => [3, 1], "test_indices" => [2, 4])
    split = ACEUtils.validate_split_indices(good, 4, 2, 2)
    @test split.train_indices == [3, 1]
    @test split.test_indices == [2, 4]
    @test_throws ArgumentError ACEUtils.validate_split_indices(
        Dict("train_indices" => [1, 1], "test_indices" => [2, 4]), 4, 2, 2)
    @test_throws ArgumentError ACEUtils.validate_split_indices(
        Dict("train_indices" => [1, 3], "test_indices" => [3, 4]), 4, 2, 2)
    @test_throws ArgumentError ACEUtils.validate_split_indices(
        Dict("train_indices" => [0, 1], "test_indices" => [2, 4]), 4, 2, 2)
end

@testset "Saved split provenance and isolated random state" begin
    mktempdir() do dir
        source, destination = joinpath(dir, "source.xyz"), joinpath(dir, "split.toml")
        write(source, "Synthetic split-integrity fixture; no atomic loading is performed.\n")
        config = Dict("dataset" => Dict("file" => source, "split_file" => destination,
            "split_seed" => 42, "expected_train" => 3, "expected_test" => 1))
        ACEUtils.Random.seed!(19)
        expected_random = rand(4)
        ACEUtils.Random.seed!(19)
        first = ACEUtils.load_or_create_split(config, 4)
        @test rand(4) == expected_random
        @test isfile(destination)
        @test first.provenance["julia_version"] == string(VERSION)
        @test first.provenance["source_sha256"] == ACEUtils.file_sha256(source)
        second = ACEUtils.load_or_create_split(config, 4)
        @test first.train_indices == second.train_indices
        @test first.test_indices == second.test_indices
        write(source, "Modified input must invalidate the saved split.\n")
        @test_throws ArgumentError ACEUtils.load_or_create_split(config, 4)
    end
end

@testset "Per-frame case-insensitive labels and original assembly row ranges" begin
    mktempdir() do dir
        path = joinpath(dir, "tiny.xyz")
        write(path, """
        1
        Lattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3:dft_force:R:3 dft_energy=-1.0 pbc="T T T"
        Si 0 0 0 0.1 0.2 0.3
        2
        Lattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3:DFT_force:R:3 DFT_energy=-2.0 pbc="T T T"
        Si 0 0 0 0.4 0.5 0.6
        Si 2 0 0 -0.4 -0.5 -0.6
        """)
        systems = ACEUtils.ExtXYZ.load(path)
        data = [ACEUtils.ACEpotentials.AtomsData(system;
            energy_key="dft_energy", force_key="dft_force") for system in systems]
        @test all(d -> !isnothing(d.energy_key) && !isnothing(d.force_key), data)
        info = ACEUtils.get_data_indices(data)
        @test info.E_inds == [5, 1]
        @test info.F_inds == vcat(6:11, 2:4)
        @test info.N_atoms_E == [2, 1]
        @test info.W_E ≈ [1 / sqrt(2), 1.0]
        @test info.row_count == 11
        @test vcat(ACEUtils.ACEfit.target_vector.(data)...)[info.E_inds] == [-2.0, -1.0]
        @test_throws ArgumentError ACEUtils.make_labeled_data(systems, nothing,
            (energy_key="missing_energy", force_key="dft_force", virial_key=nothing))
        @test_throws ArgumentError ACEUtils.get_data_indices(ACEUtils.ACEpotentials.AtomsData[])
    end
end
