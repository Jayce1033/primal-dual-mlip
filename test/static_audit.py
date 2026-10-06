#!/usr/bin/env python3
"""Read-only source, configuration, and XYZ audit; never fits an ACE model.

Python >=3.9 is sufficient. TOML and saved-split checks additionally use the
standard-library tomllib when Python >=3.11 is available.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

try:
    import tomllib
except ImportError:
    tomllib = None


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "LiCl/LiCl_E0s.xyz": (2, 2, "8cfb1d97bbdb4d07c53afe8f58215436cd58ef8f67ee2018a8cb48a97541170e", {"energy": 2}, {"forces": 2}),
    "LiCl/LiCl_test.xyz": (52, 3496, "db220516a02e0f6d8578884964fc6b30ff27da9422a0ecc5392a714abb3ea551", {"energy": 52}, {"forces": 52}),
    "LiCl/LiCl_train.xyz": (586, 38234, "213d14998d88569ded47304cb7360ee2760738b408f65d3ee8f10b28d4d532fa", {"energy": 586}, {"forces": 586}),
    "OOD-Si/E0s-Si.xyz": (1, 1, "a6e8698a174674cebe7d79d22d953da044194084d8a03dfe5b42f6cd60a93641", {"dft_energy": 1}, {"dft_force": 1}),
    "OOD-Si/Si_test.xyz": (28, 1020, "5b5d9152488a4527211e4b9b5c942b1852f66fad8972dc2bd97690fbfcc06aa3", {"energy": 28}, {"dft_force": 28}),
    "OOD-Si/Si_test_filtered.xyz": (21, 576, "eaf5a112651228c3f4387c5f5349612d7479c3c44adb488d586869d48803f460", {"energy": 21}, {"dft_force": 21}),
    "OOD-Si/Si_train.xyz": (2475, 171815, "74ac5f387aa0149ec9ea8f3e275d1087d7287db68c15ba8bc4c28a450de67e2a", {"dft_energy": 2415, "DFT_energy": 60}, {"dft_force": 2415, "DFT_force": 60}),
    "water/dataset_1593.xyz": (1593, 305856, "5e2ec3b4b1334e012daa6a2e2dbd138108c41c7b6f692ff89b8747b81cadfdbc", {"TotEnergy": 1593}, {"force": 1593}),
}
HEADER_FIELD = re.compile(r'(?:^|\s)([A-Za-z_][A-Za-z_0-9]*)=(?:"([^"]*)"|(\S+))')
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def scan_xyz(path, energy_keys, force_keys):
    """Stream bytes once for checksums, structural validation, and label counts."""
    digest = hashlib.sha256()
    frames = atoms = components = 0
    energies, forces, config_types = Counter(), Counter(), Counter()
    with path.open("rb") as stream:
        def read():
            line = stream.readline()
            digest.update(line)
            return line

        while True:
            raw = read()
            if not raw:
                break
            if not raw.strip():
                continue
            natoms = int(raw)
            if natoms < 1:
                raise ValueError("nonpositive atom count")
            header = read().decode("utf-8")
            fields = {m.group(1): m.group(2) if m.group(2) is not None else m.group(3)
                      for m in HEADER_FIELD.finditer(header)}
            properties = fields.get("Properties", "").split(":")
            if not properties or len(properties) % 3:
                raise ValueError("missing or malformed Properties field")
            columns, offset = {}, 0
            for name, kind, width in zip(properties[::3], properties[1::3], properties[2::3]):
                size = int(width)
                if size < 1 or kind not in {"R", "I", "L", "S"}:
                    raise ValueError("invalid XYZ property definition")
                columns[name] = (offset, size)
                offset += size
            selected_energy = [key for key in energy_keys if key in fields]
            selected_force = [key for key in force_keys if key in columns]
            if len(selected_energy) != 1 or len(selected_force) != 1:
                raise ValueError("frame {}: missing or ambiguous target labels".format(frames + 1))
            if not math.isfinite(float(fields[selected_energy[0]])):
                raise ValueError("non-finite energy")
            force_start, force_width = columns[selected_force[0]]
            if force_width != 3:
                raise ValueError("force property must have three scalar components")
            for _ in range(natoms):
                values = read().decode("utf-8").split()
                if len(values) != offset:
                    raise ValueError("frame {}: truncated or malformed atom row".format(frames + 1))
                if not all(math.isfinite(float(x)) for x in values[force_start:force_start + 3]):
                    raise ValueError("non-finite force")
            energies.update(selected_energy)
            forces.update(selected_force)
            config_types.update([fields.get("config_type", "<missing>")])
            frames += 1
            atoms += natoms
            components += 3 * natoms
    return {
        "frames": frames, "atoms": atoms, "force_components": components,
        "energy_labels": dict(energies), "force_labels": dict(forces),
        "config_types": dict(config_types), "sha256": digest.hexdigest(),
    }


def audit_data(failures):
    records = {}
    actual_paths = {str(p.relative_to(ROOT / "data")) for p in (ROOT / "data").rglob("*.xyz")}
    if actual_paths != set(EXPECTED):
        failures.append("XYZ inventory differs: {}".format(sorted(actual_paths ^ set(EXPECTED))))
    for relative, (frames, atoms, checksum, energies, forces) in EXPECTED.items():
        try:
            record = scan_xyz(ROOT / "data" / relative, energies, forces)
            records[relative] = record
            for key, value in [("frames", frames), ("atoms", atoms), ("sha256", checksum),
                               ("energy_labels", energies), ("force_labels", forces)]:
                if record[key] != value:
                    failures.append("{}: unexpected {}: {}".format(relative, key, record[key]))
        except (OSError, ValueError, UnicodeError) as error:
            failures.append("{}: {}".format(relative, error))
    return records


def audit_sources(failures):
    required = ["README.md", "Project.toml", "Manifest.toml", "CITATION.cff", ".gitignore",
                "src/ACEUtils.jl", "src/train_model.jl", "data/README.md", "results/README.md",
                "scripts/train.jl", "scripts/run_scan.jl", "scripts/run_ablation.jl",
                "scripts/run_warmstart.jl", "scripts/analyze_pareto.jl", "scripts/prepare_data.jl"]
    for relative in required:
        if not (ROOT / relative).is_file():
            failures.append("required file missing: {}".format(relative))
    for relative in ["src/solvers", "configs/datasets", "configs/experiments", "test", "data/splits"]:
        if not (ROOT / relative).is_dir():
            failures.append("required directory missing: {}".format(relative))
    pd_definitions = []
    for path in sorted(ROOT.rglob("*.jl")):
        if "results" in path.relative_to(ROOT).parts:
            continue
        source = path.read_text(encoding="utf-8")
        local_modules = re.findall(r"(?:using|import)\s+\.([A-Za-z_][A-Za-z_0-9]*)", source)
        if any(name != "ACEUtils" for name in local_modules):
            failures.append("noncanonical module reference: {}".format(path.relative_to(ROOT)))
        for number, line in enumerate(source.splitlines(), 1):
            if CJK.search(line):
                failures.append("non-English source text: {}:{}".format(path.relative_to(ROOT), number))
        pd_definitions.extend(re.findall(r"(?m)^\s*function\s+(solve_pd\w*)\s*\(", source))
    if pd_definitions != ["solve_pd2_hpc_complete1"]:
        failures.append("expected exactly the canonical PD implementation; found {}".format(pd_definitions))
    if (ROOT / "ACEUtils.jl").exists():
        failures.append("duplicate module entry point at repository root")


def inspect_split_tables(node, path, failures, checked):
    if not isinstance(node, dict):
        return
    if "train_indices" in node or "test_indices" in node:
        train, test = node.get("train_indices"), node.get("test_indices")
        if not isinstance(train, list) or not isinstance(test, list):
            failures.append("{}: split indices must both be arrays".format(path))
            return
        indices = train + test
        if (len(train) != 1274 or len(test) != 319 or
                any(type(i) is not int for i in indices) or
                sorted(indices) != list(range(1, 1594))):
            failures.append("{}: invalid water split sizes, duplicates, or coverage".format(path))
        for key, expected in [("seed", 42), ("n_total", 1593),
                              ("algorithm", "Random.seed! + randperm"),
                              ("source_sha256", EXPECTED["water/dataset_1593.xyz"][2])]:
            if node.get(key) != expected:
                failures.append("{}: missing or mismatched split provenance {}".format(path, key))
        if not isinstance(node.get("julia_version"), str) or not node["julia_version"]:
            failures.append("{}: generating Julia version is missing".format(path))
        checked.append(str(path))
    for value in node.values():
        if isinstance(value, dict):
            inspect_split_tables(value, path, failures, checked)


def audit_toml(failures, notices):
    if tomllib is None:
        notices.append("SKIP TOML parsing and saved-split validation: Python >=3.11 is required.")
        return
    checked = []
    for path in sorted(ROOT.rglob("*.toml")):
        if "results" in path.relative_to(ROOT).parts:
            continue
        try:
            with path.open("rb") as stream:
                parsed = tomllib.load(stream)
            if "splits" in path.relative_to(ROOT).parts:
                inspect_split_tables(parsed, path.relative_to(ROOT), failures, checked)
        except (OSError, ValueError) as error:
            failures.append("{}: {}".format(path.relative_to(ROOT), error))
    if not checked:
        notices.append("PENDING water split: run the Julia data-preparation script and commit its indices.")
    else:
        notices.append("Validated saved split coverage: {}".format(", ".join(checked)))


def audit_ignore(failures, notices):
    if not (ROOT / ".git").exists():
        notices.append("SKIP Git ignore check: this directory is not yet an initialized Git repository.")
        return
    ignored = ["results/audit-example/model.json", "results/audit-example/model.bson", "results/audit-example/log.csv"]
    tracked = ["results/README.md", "data/splits/audit-example.json", "configs/datasets/audit-example.json", "Manifest.toml"]
    for path in ignored + tracked:
        try:
            result = subprocess.run(["git", "check-ignore", "--no-index", "--quiet", path], cwd=ROOT,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        except OSError as error:
            failures.append("Git ignore check failed: {}".format(error))
            return
        if result.returncode not in {0, 1}:
            failures.append("Git check-ignore failed for {}".format(path))
        elif (result.returncode == 0) != (path in ignored):
            failures.append("unexpected Git ignore status for {}".format(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-only", action="store_true", help="only inspect raw XYZ data")
    parser.add_argument("--json", action="store_true", help="print machine-readable results to stdout")
    args = parser.parse_args()
    failures, notices = [], []
    records = audit_data(failures)
    if not args.data_only:
        audit_sources(failures)
        audit_toml(failures, notices)
        audit_ignore(failures, notices)
    if args.json:
        print(json.dumps({"data": records, "failures": failures, "notices": notices}, indent=2))
    else:
        for relative, record in records.items():
            print("{}: {} frames; {} force components; sha256={}".format(
                relative, record["frames"], record["force_components"], record["sha256"]))
        for notice in notices:
            print(notice)
        for failure in failures:
            print("FAIL " + failure, file=sys.stderr)
        print("Static audit: {} ({} errors). No Julia code or training was executed.".format(
            "PASS" if not failures else "FAIL", len(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
