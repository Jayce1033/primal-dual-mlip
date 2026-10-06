# Reference data

Training uses the following Extended XYZ inputs. Each configuration supplies
one total energy and three force components per atom; virials are not fitted.
Input positions, energies, and forces use Angstrom, eV, and eV/Angstrom,
respectively. The loader performs no numerical unit conversion.

| System | Training input | Test input | Training / test configurations |
| --- | --- | --- | ---: |
| LiCl | `LiCl/LiCl_train.xyz` | `LiCl/LiCl_test.xyz` | 586 / 52 |
| Water | `water/dataset_1593.xyz` | Same file, saved partition | 1274 / 319 |
| Si | `OOD-Si/Si_train.xyz` | `OOD-Si/Si_test_filtered.xyz` | 2475 / 21 |

Paths are relative to this directory. LiCl labels are `energy` and `forces`;
water labels are `TotEnergy` and `force`. Si training uses `dft_energy` and
`dft_force`, including case variants; testing uses `energy` and `dft_force`.

LiCl training includes two isolated-atom configurations. Si training contains
crystalline and defect structures together with 76 `liq` and 159 `amorph`
configurations; its test input contains 20 `SF` and one `aSi` configuration.
The loader uses these files directly without filtering their structure classes.

The isolated-atom files `LiCl/LiCl_E0s.xyz` and `OOD-Si/E0s-Si.xyz`, and the
unfiltered `OOD-Si/Si_test.xyz`, are auxiliary inputs and are not appended to
training. Si uses a reference energy of `-158.54496821` eV per atom.

Water contains 1593 configurations of 192 atoms each. Its partition is generated
with Julia seed 42 and persisted as described in [splits/README.md](splits/README.md).
The training RNG seed is configured independently.

## Source studies

The dataset attributions are:

- LiCl: Sivaraman et al., *Automated Development of Molten Salt Machine Learning
  Potentials: Application to LiCl* (2021),
  [DOI: 10.1021/acs.jpclett.1c00901](https://doi.org/10.1021/acs.jpclett.1c00901).
- Water: Cheng et al., *Ab initio thermodynamics of liquid and solid water*
  (2019), [DOI: 10.1073/pnas.1815117116](https://doi.org/10.1073/pnas.1815117116).
  The fitting data are attributed to revPBE0-D3 liquid water.
- Si: Bartok et al., *Machine Learning a General-Purpose Interatomic Potential
  for Silicon* (2018),
  [DOI: 10.1103/PhysRevX.8.041048](https://doi.org/10.1103/PhysRevX.8.041048).

These references identify the source studies. Exact supplied files are identified
by the SHA-256 inventory in `test/static_audit.py`; the repository does not assign
a new license to third-party data.

Run `python3 test/static_audit.py` from the repository root to verify every
input file's hash, frame layout, label availability, and finite target values.
Keep raw inputs and saved splits in version control; store generated models,
matrices, logs, and trajectories under `results/`.
