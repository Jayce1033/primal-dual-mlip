# Saved dataset splits

Water uses Julia `Random.seed!(42)` followed by `randperm(1593)`. The first
1274 indices select training configurations and the remaining 319 select
testing configurations. Indices are one-based and preserve permutation order.

Run `julia --project=. scripts/prepare_data.jl --dataset water` from the repository
root to generate `water_seed42.toml`. The file records the indices, source
SHA-256, Julia version, seed, and algorithm. Generation runs in an isolated
Julia Task without reseeding the training task. Preserve this file with each
experiment, since a seed alone does not fix a permutation across RNG versions.

Subsequent loads verify the source hash, seed, bounds, sizes, uniqueness,
disjointness, and complete coverage before reusing the saved indices. A changed
source or seed produces an error rather than regenerating an existing split.

LiCl and Si use separate training and test files without a random partition.
