# Generated results

Training creates a separate directory for each run containing:

| File | Contents |
| --- | --- |
| `resolved_config.toml` | Effective run configuration |
| `model.json` | ACE model with final coefficients |
| `summary.json` | Metrics, solver status, stage timings, and input/environment provenance |
| `training.bson` | Coefficients, multiplier, stage histories, and PD optimizer states |

Multi-run experiments also write `scan.json`, `ablation.json`, or
`warmstart.json`. Retain the resolved settings, input hashes, split indices,
runtime versions, and thread counts when sharing an experiment.

Generated outputs in this directory are ignored by Git except for this README.
