# primal-dual-mlip

Julia code for force-constrained fitting of fixed-basis atomic cluster expansion
(ACE) models, accompanying **From Energy--Force Weighting to Primal--Dual
Optimization of Machine-Learned Interatomic Potentials**, by Chenyu Wang,
Yangshuai Wang, and Lei Zhang.

Paper: [arXiv:2610.00876](https://arxiv.org/abs/2610.00876).

The repository includes LiCl, liquid-water, and Si inputs, a shared training
interface, primal-dual optimization with fixed-multiplier L-BFGS refinement,
static and LSQR comparisons, weight scans, ablations, and warm starts.

## Installation

Use Julia **1.11.5** with the supplied `Project.toml` and `Manifest.toml`.
The environment pins ACEpotentials **0.9.1**, Optim **1.13.3**, and Optimisers
**0.3.4**. Add the required registries once on a fresh Julia installation:

```julia
using Pkg
Pkg.Registry.add("General")
Pkg.Registry.add(Pkg.RegistrySpec(url="https://github.com/ACEsuit/ACEregistry"))
```

Install the dependencies from the repository root:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Training

Settings merge in this order: dataset defaults, experiment configuration, then
explicit `--set section.key=value` overrides. Use `--dataset licl|water|si` to
select a dataset. Configuration paths are repository-relative.

```sh
julia --project=. scripts/train.jl --config configs/experiments/pd_licl.toml --check
```

`--check` validates the selected configuration without loading data or assembling
matrices. Run the same command without `--check` to fit the model. The three main
PD configurations specify all training parameters. Comparison templates expose
`"REQUIRED"` fields for user-selected weights and experiment settings.
Each entry point supports `--help`.

| Task | Script | Configuration in `configs/experiments/` |
| --- | --- | --- |
| PD and L-BFGS refinement | `train.jl` | `pd_licl.toml`, `pd_water.toml`, `pd_si.toml` |
| Static-weight L-BFGS | `train.jl` | `static.toml` |
| LSQR fit | `train.jl` | `lsqr.toml` |
| Heuristic-weight baseline | `train.jl` | `heuristic.toml` |
| Weight scan | `run_scan.jl` | `static.toml` |
| LiCl ablations | `run_ablation.jl` | `ablation.toml` |
| LiCl warm starts | `run_warmstart.jl` | `warmstart.toml` |
| Pareto analysis | `analyze_pareto.jl` | Existing scan output |

Scripts are located in `scripts/`. Scans accept `--solver lbfgs|lsqr` and
`--base-grid` for the nine-point logit grid without additional knee weights.
Ablations accept `--sweep force_batch|primal_lr|dual_lr|all`.

The programmatic interface uses the same configuration:

```julia
include("src/ACEUtils.jl")
using .ACEUtils

config = load_config("configs/experiments/pd_licl.toml")
validate_config(config)
result = train_model(config)
```

Importing `ACEUtils` does not load datasets or start training. `batch_size_E`
counts configuration energies; `batch_size_F` counts scalar force components.
The streams are sampled independently, and `"all"` selects all observations of
one type. The multiplier is projected onto `[0, pd.lambda_F_max]`, with a default
upper bound of `1e4`. Refinement uses the final PD coefficients and multiplier.
Set `primal_lr` and `initial_learning_rate` together when changing the primal
learning rate.

## Data and outputs

See [data/README.md](data/README.md) for input files, labels, units, and source
studies. Generate the water partition before training:

```sh
julia --project=. scripts/prepare_data.jl --dataset water
```

This writes the 1274/319 partition to `data/splits/water_seed42.toml`, recording
the actual indices, source hash, and Julia version. Subsequent runs reuse it.

Each fit writes `resolved_config.toml`, `model.json`, `summary.json`, and
`training.bson` beneath its configured `output_dir`. Scans, ablations, and warm
starts also write aggregate JSON reports. Generated files under `results/`
are ignored by Git.

`runtime.blas_threads` defaults to 1; Julia threads are selected at process
launch. Outputs record both thread counts and separate assembly, optimization,
and evaluation timings.

Citation metadata are in `CITATION.cff`. Third-party data retain their
respective terms.
