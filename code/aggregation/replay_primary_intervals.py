"""Replay primary conditional intervals from canonical labels and fixed predictions.

The numerical functions and Table 2 assembly are taken unchanged from the
released aggregation scripts. Only input loading, output paths and the OpenBLAS kernel setting are adapted.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import os

# Trailing digits of the bootstrap dot products depend on the OpenBLAS kernel; a value set in the shell wins.
os.environ.setdefault("OPENBLAS_CORETYPE", "Prescott")

import numpy as np
import pandas as pd

import replay_main_tables as io

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def load_pairs(inputs=None, pairs_path=None, pair_strata=None):
    """Read either a prepared manifest or an explicit canonical pair table."""
    io.require((inputs is None) != (pairs_path is None), "Provide exactly one of --inputs or --pairs")
    dependency_path = REPO / "code/source_components/mappings/pair_dependency_components.csv"
    dependency = pd.read_csv(dependency_path)
    fold_path = REPO / "code/data_build/curation/outer_identity_folds.csv"
    bound_paths = [dependency_path]
    if inputs is not None:
        files = io.read_inputs(Path(inputs))
        records, pairs, folds = (pd.read_csv(files[key]) for key in ("records", "pairs", "outer_folds"))
        io.validate_inputs(records, pairs, folds, dependency)
        bound_paths.extend([Path(inputs).resolve(), *files.values()])
    else:
        pairs_path = Path(pairs_path).resolve(strict=True)
        pairs = pd.read_csv(pairs_path)
        folds = pd.read_csv(fold_path)
        index = pd.read_csv(REPO / "code/data_build/curation/pair_index.csv").set_index("pair_id")
        io.require(pairs.pair_id.is_unique and set(pairs.pair_id) == set(index.index),
                   "Canonical pair population differs from the public pair index")
        io.same_keys(pairs, index, "pair_id", ["record_id_i", "record_id_j"], "Canonical pairs")
        io.require("canonical_target_b_minus_a" in pairs
                   and np.isfinite(pairs.canonical_target_b_minus_a.to_numpy(float)).all(),
                   "Finite canonical_target_b_minus_a labels are required")
        bound_paths.extend([pairs_path, fold_path])
    pairs = io.attach_strata(pairs, pair_strata)
    if pair_strata is not None:
        bound_paths.append(Path(pair_strata).resolve(strict=True))
    return pairs, folds, dependency, bound_paths


def original_assembly(original, pred, write_csv):
    """Execute exact original AST statements; do not translate their arithmetic."""
    source_path = HERE / "run_numerics_bootstrap.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    statements = [
        "repeat,aggregate,primary,rep_compare,gamma,estimates=original.summarize_metrics(pred)",
        "draws,intervals=original.bootstrap(pred,estimates)",
        "csv('conditional_component_bootstrap.csv',draws)",
        "csv('primary_estimates.csv',intervals)",
        "csv('table2_contrasts.csv',intervals.loc[~intervals.statistic.str.startswith('log_C1_over_direct')])",
        "csv('table_s8_contrasts.csv',intervals)",
    ]
    selected = []
    for statement in statements:
        target = ast.dump(ast.parse(statement).body[0], include_attributes=False)
        matches = [node for node in main.body if ast.dump(node, include_attributes=False) == target]
        io.require(len(matches) == 1, "Original primary-interval assembly statement missing or ambiguous")
        selected.append(matches[0])
    io.require([node.lineno for node in selected] == sorted(node.lineno for node in selected),
               "Original primary-interval assembly order changed")
    namespace = {"original": original, "pred": pred, "csv": write_csv}
    program = ast.Module(body=selected, type_ignores=[])
    exec(compile(program, str(source_path), "exec"), namespace)
    evidence = [{"line": node.lineno, "end_line": node.end_lineno,
                 "source": ast.get_source_segment(source, node)} for node in selected]
    return namespace["draws"], namespace["intervals"], evidence


def replay(pairs, folds, dependency, output, bound_paths=()):
    """Run the fixed protocol once; callers may supply any explicit pair labels."""
    output = Path(output).resolve()
    io.require(not output.exists(), "Output directory must not already exist")
    io.require(not output.is_relative_to(REPO), "Output must be outside the repository")
    original_path = HERE / "summarize_r5.py"
    original = io.load_module(original_path, "primary_intervals_original_summary")
    io.require(original.DRAWS == 2000 and original.SEED == 20261027, "Original bootstrap protocol drift")
    prediction_path = REPO / "results/predictions/current_companion/S06_outer_predictions_long.csv.gz"
    pred = io.read_internal(prediction_path, pairs, folds, dependency, original)
    components = sorted(pred.dependency_component.unique())
    io.require(len(components) == 38, "Expected 38 tested dependency components")
    output.mkdir(parents=True)

    def write_csv(name, frame):
        target = "S08_primary_estimates.csv" if name == "primary_estimates.csv" else name
        frame.to_csv(output / target, index=False, lineterminator="\n")

    print("Executing original primary summaries and 2000-draw component bootstrap.", flush=True)
    draws, intervals, assembly = original_assembly(original, pred, write_csv)
    bindings = [*bound_paths, prediction_path, original_path, HERE / "run_numerics_bootstrap.py",
                HERE / "replay_main_tables.py", Path(__file__).resolve()]
    receipt = {
        "status": "PRIMARY_CONDITIONAL_INTERVAL_REPLAY_COMPLETE",
        "prediction_rows": len(pred), "pairs": len(pairs),
        "model_cells": len(pred.groupby(["representation", "learner", "repeat", "pipeline"])),
        "components_in_original_sorted_order": components,
        "component_count": len(components), "draws": len(draws), "seed": original.SEED,
        "statistics": len(intervals), "valid_draws": intervals.valid_draws.tolist(),
        "invalid_draws": intervals.invalid_draws.tolist(),
        "fits": 0, "model_loads": 0, "new_predictions": 0, "bootstrap_calls": 1,
        "parsing": "pandas default for canonical labels; round_trip for released internal predictions",
        "assembly": assembly,
        "output_name_mapping": {"primary_estimates.csv": "S08_primary_estimates.csv"},
        "input_bindings": {str(path): io.digest(path) for path in bindings},
        "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
        "scope": "Fixed selected fitted predictions; conditional component-resampling intervals only",
    }
    (output / "REPLAY_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": receipt["status"], "output": str(output),
                      "statistics": len(intervals), "draws": len(draws),
                      "component_count": len(components), "seed": original.SEED,
                      "bootstrap_calls": 1, "fits": 0, "new_predictions": 0}))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--inputs", type=Path, help="Reconstruction inputs.json")
    source.add_argument("--pairs", type=Path, help="Pair CSV with canonical_target_b_minus_a in j-minus-i direction")
    parser.add_argument("--pair-strata", type=Path,
                        help="Curated pair_id,time_status,any_approximate_label CSV, if absent from pairs")
    parser.add_argument("--output", type=Path, required=True, help="New output directory outside this repository")
    args = parser.parse_args()
    pairs, folds, dependency, bindings = load_pairs(args.inputs, args.pairs, args.pair_strata)
    replay(pairs, folds, dependency, args.output, bindings)


if __name__ == "__main__":
    main()
