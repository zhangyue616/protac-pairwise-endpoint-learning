"""Replay 14 main-data tables from reconstructed inputs and released predictions.

Calls the archived statistical functions without running their historical main
programs. No estimator, bootstrap, source-case or historical-table input is used.
The two stratification fields come from the supplied pair_strata.csv or equivalent curation, never from result tables.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STRATA = ("time_status", "any_approximate_label")


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_inputs(path):
    path = path.resolve(strict=True)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == "PREPARED_CURATED_INPUTS_V1",
            "Expected reconstruction inputs.json (PREPARED_CURATED_INPUTS_V1)")
    files = {}
    for key in ("records", "pairs", "outer_folds"):
        require(key in manifest.get("files", {}), "Missing prepared input: " + key)
        item = manifest["files"][key]
        target = (path.parent / item["path"]).resolve(strict=True)
        require(target.is_relative_to(path.parent), "Input path escapes prepared directory: " + key)
        require(digest(target) == item["sha256"].upper(), "Prepared input SHA mismatch: " + key)
        files[key] = target
    return files


def same_keys(frame, authority, key, fields, label):
    require(not frame[key].isna().any() and set(frame[key]) <= set(authority.index),
            label + ": unknown/missing " + key)
    aligned = authority.loc[frame[key]]
    for field in fields:
        require(field in frame and field in aligned, label + ": missing " + field)
        require(np.array_equal(frame[field].to_numpy(), aligned[field].to_numpy()),
                label + ": identity/direction mismatch: " + field)


def validate_inputs(records, pairs, folds, dependency):
    require(len(records) == 699 and records.record_id.is_unique
            and records.identity_hash.nunique() == 511, "Record/identity population drift")
    require(len(pairs) == 5804 and pairs.pair_id.is_unique, "Expected 5804 unique reconstructed pairs")
    require(records.record_id.notna().all(), "Missing record identifier")
    record_index = pd.read_csv(REPO / "code/data_build/curation/record_index.csv").set_index("record_id")
    require(set(records.record_id) == set(record_index.index), "Record keys differ from public index")
    fields = ["identity_hash", "normalized_poi_id", "pair_context_hash"]
    same_keys(records, record_index, "record_id", fields, "Reconstructed records")
    pair_index = pd.read_csv(REPO / "code/data_build/curation/pair_index.csv").set_index("pair_id")
    require(set(pairs.pair_id) == set(pair_index.index), "Pair keys differ from public index")
    same_keys(pairs, pair_index, "pair_id", ["record_id_i", "record_id_j"], "Reconstructed pairs")
    by_record = records.set_index("record_id")
    for side in ("i", "j"):
        require(set(pairs["record_id_" + side]) <= set(by_record.index), "Unknown pair endpoint")
        mapped = by_record.loc[pairs["record_id_" + side]]
        require(np.array_equal(pairs["identity_hash_" + side].to_numpy(), mapped.identity_hash.to_numpy()),
                "Reconstructed pair identity differs: " + side)
        for field in ("normalized_poi_id", "pair_context_hash"):
            require(np.array_equal(pairs[field].to_numpy(), mapped[field].to_numpy()),
                    "Reconstructed pair context differs: " + field)
    require("canonical_target_b_minus_a" in pairs, "Reconstructed canonical forward target missing")
    target = pairs.canonical_target_b_minus_a.to_numpy(float)
    require(np.isfinite(target).all(), "Nonfinite reconstructed canonical target")
    forward = (pairs.record_id_j.map(by_record.pdc50_median)
               - pairs.record_id_i.map(by_record.pdc50_median)).to_numpy(float)
    require(np.array_equal(np.sign(target), np.sign(forward))
            and np.array_equal(target == 0, forward == 0),
            "Canonical target direction/zero state differs from pDC50(j)-pDC50(i)")
    require(np.array_equal(pairs.true_delta.to_numpy(float), -target),
            "Reconstructed historical true_delta is not exactly negative canonical target")
    require(len(dependency) == len(pairs) and dependency.pair_id.is_unique
            and set(dependency.pair_id) == set(pairs.pair_id), "Public pair/component scope mismatch")
    same_keys(dependency, pairs.set_index("pair_id"), "pair_id", ["record_id_i", "record_id_j"],
              "Public dependency mapping")
    require(dependency.dependency_component.notna().all()
            and dependency.dependency_component.nunique() == 43, "Component population drift")
    require(set(folds["repeat"]) == {0, 1, 2}, "Outer repeat population drift")
    for repeat in range(3):
        part = folds.loc[folds["repeat"].eq(repeat)]
        require(part.identity_hash.is_unique and set(part.identity_hash) == set(records.identity_hash)
                and set(part.outer_fold) == set(range(5)), "Outer fold identity coverage drift")
    frozen_folds = pd.read_csv(REPO / "code/data_build/curation/outer_identity_folds.csv")
    keys = ["repeat", "identity_hash"]
    require(folds.sort_values(keys).reset_index(drop=True)[keys + ["outer_fold"]].equals(
        frozen_folds.sort_values(keys).reset_index(drop=True)[keys + ["outer_fold"]]),
        "Prepared outer assignments differ from public frozen assignments")


def attach_strata(pairs, path):
    result = pairs.copy()
    if path is not None:
        strata = pd.read_csv(path)
        require(set(strata.columns) == {"pair_id", *STRATA},
                "--pair-strata must contain only pair_id,time_status,any_approximate_label")
        require(strata.pair_id.is_unique and len(strata) == len(pairs)
                and set(strata.pair_id) == set(pairs.pair_id), "Pair-strata population differs")
        aligned = strata.set_index("pair_id").loc[pairs.pair_id]
        for field in STRATA:
            if field in result:
                require(np.array_equal(result[field].to_numpy(), aligned[field].to_numpy()),
                        "Pair-strata conflicts with reconstructed pairs: " + field)
            else:
                result[field] = aligned[field].to_numpy()
    require(set(STRATA) <= set(result),
            "Reconstruction lacks time_status/any_approximate_label; provide --pair-strata code/aggregation/pair_strata.csv")
    require(result.time_status.notna().all() and set(result.time_status) <= {
        "same_recorded_fixed_time", "recorded_context_time_unknown", "common_reported_time_window"},
        "Unknown/missing curated time_status")
    require(result.any_approximate_label.notna().all()
            and result.any_approximate_label.isin([True, False]).all(), "Invalid approximate-label boolean")
    return result


def read_internal(path, pairs, folds, dependency, original):
    # The released long table holds the study's values as text; labels keep their own default parse.
    frame = pd.read_csv(path, float_precision="round_trip")
    require("true_delta" not in frame, "Released prediction input unexpectedly includes target labels")
    same_keys(frame, pairs.set_index("pair_id"), "pair_id",
              ["record_id_i", "record_id_j", "identity_hash_i", "identity_hash_j", "pair_context_hash"],
              "Internal predictions")
    same_keys(frame, dependency.set_index("pair_id"), "pair_id", ["dependency_component"],
              "Internal prediction components")
    require(frame.g_seen_in_outer_train.notna().all()
            and frame.g_seen_in_outer_train.isin([True, False]).all(), "Invalid g_seen boolean")
    key = ["representation", "learner", "repeat", "outer_fold", "pipeline", "pair_id"]
    require(not frame.duplicated(key).any(), "Duplicate internal prediction key")
    grid = {(rep, learner, repeat, fold, arm) for rep in original.REPS
            for learner in original.LEARNERS for repeat in range(3) for fold in range(5)
            for arm in original.arms_for(learner)}
    require(set(map(tuple, frame[key[:-1]].drop_duplicates().to_numpy())) == grid,
            "Incomplete/extra internal model grid")
    expected = {}
    for repeat in range(3):
        assignment = folds.loc[folds["repeat"].eq(repeat)].set_index("identity_hash").outer_fold
        left, right = pairs.identity_hash_i.map(assignment), pairs.identity_hash_j.map(assignment)
        for fold in range(5):
            expected[repeat, fold] = set(pairs.loc[left.eq(fold) & right.eq(fold), "pair_id"])
    for (_, _, repeat, fold, _), part in frame.groupby(key[:-1], sort=True):
        require(set(part.pair_id) == expected[repeat, fold], "Internal test-pair support mismatch")
    numeric = frame[["raw_prediction", "scale_c", "calibrated_prediction"]].to_numpy(float)
    require(np.isfinite(numeric).all() and frame.scale_c.ge(0).all(), "Nonfinite/negative calibration")
    require(np.allclose(numeric[:, 0] * numeric[:, 1], numeric[:, 2], rtol=original.TOL, atol=original.TOL),
            "Internal calibration mismatch")
    index = pairs.set_index("pair_id")
    frame["true_delta"] = frame.pair_id.map(index.canonical_target_b_minus_a)
    for field in STRATA:
        frame[field] = frame.pair_id.map(index[field])
    return frame


def read_component(path, pairs, dependency, census, support, aggregate):
    # Historical aggregate_full_r6.read_csv uses round_trip on producer CSVs.
    frame = pd.read_csv(path, float_precision="round_trip")
    require("true_delta" not in frame, "Released prediction input unexpectedly includes target labels")
    index = pairs.set_index("pair_id")
    frame["true_delta"] = frame.pair_id.map(index.canonical_target_b_minus_a)
    pair_map = pairs[["pair_id", "record_id_i", "record_id_j"]].copy()
    pair_map["heldout_component_id"] = pair_map.pair_id.map(
        dependency.set_index("pair_id").dependency_component)
    authority = pair_map.copy()
    authority["true_delta"] = pairs.canonical_target_b_minus_a.to_numpy()
    plan = {"learners": ["ridge", "random_forest", "xgboost"],
            "representations": ["binary", "count"],
            "pipelines": ["direct_pair", "endpoint_E0", "endpoint_E0_matched", "endpoint_E1"]}
    aggregate.validate_predictions(frame, authority.set_index("pair_id"), plan)
    counts = pair_map.groupby("heldout_component_id").size()
    selected = census.loc[census.pairs.gt(0)].set_index("component_id")
    require(set(selected.index) == set(counts.index)
            and np.array_equal(selected.loc[counts.index, "pairs"].to_numpy(), counts.to_numpy()),
            "Public census pair populations differ from reconstructed pairs")
    endpoint_counts = pairs.assign(component=pair_map.heldout_component_id).groupby("component").apply(
        lambda part: len(set(part.identity_hash_i) | set(part.identity_hash_j)))
    require(np.array_equal(selected.loc[endpoint_counts.index, "pair_endpoint_identities"].to_numpy(),
                           endpoint_counts.to_numpy()), "Public census pair-endpoint identities differ")
    support_index = support.set_index("dependency_component")
    require(support.dependency_component.is_unique and set(support_index.index) == set(counts.index)
            and np.array_equal(support_index.loc[counts.index, "test_pairs"].to_numpy(), counts.to_numpy()),
            "Public support pair populations differ")
    return frame, pair_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Reconstruction inputs.json")
    parser.add_argument("--output", type=Path, required=True, help="New directory outside this repository")
    parser.add_argument("--pair-strata", type=Path,
                        help="CSV supplied at code/aggregation/pair_strata.csv: pair_id,time_status,any_approximate_label; exact pair population")
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), "Output directory must not already exist")
    require(not output.is_relative_to(REPO), "Output must be outside the repository; released results are read-only")
    files = read_inputs(args.inputs)
    records, pairs, folds = (pd.read_csv(files[key]) for key in ("records", "pairs", "outer_folds"))
    dependency_path = REPO / "code/source_components/mappings/pair_dependency_components.csv"
    dependency = pd.read_csv(dependency_path)
    validate_inputs(records, pairs, folds, dependency)
    pairs = attach_strata(pairs, args.pair_strata)
    census_path = REPO / "results/current_companion/S26_group_census.csv"
    support_path = REPO / "results/current_companion/S26_group_support.csv"
    census = pd.read_csv(census_path, float_precision="round_trip")
    support = pd.read_csv(support_path, float_precision="round_trip")
    original = load_module(HERE / "summarize_r5.py", "reader_original_r5_summary")
    # aggregate_full_r6 imports the unchanged, local prepare_corrected_metadata
    # module. Its main/prepare functions are never invoked by this wrapper.
    sys.path.insert(0, str(HERE))
    try:
        aggregate = load_module(HERE / "aggregate_full_r6.py", "reader_original_r6_aggregation")
    finally:
        sys.path.pop(0)
    prediction_root = REPO / "results/predictions/current_companion"
    internal_path = prediction_root / "S06_outer_predictions_long.csv.gz"
    component_path = prediction_root / "S29_all_corrected_source_predictions.csv.gz"
    internal = read_internal(internal_path, pairs, folds, dependency, original)
    repeat, learners, ratios, comparisons, _, _ = original.summarize_metrics(internal)
    # R6 historically parsed its source/authority CSVs with round_trip; this
    # independent read preserves that boundary rather than retuning for a match.
    component_pairs = pd.read_csv(files["pairs"], float_precision="round_trip")
    component, pair_map = read_component(component_path, component_pairs, dependency, census, support, aggregate)
    reference = HERE / "analyze_source_predictions.py"
    plan = {"analysis_definition_reference": str(reference), "analysis_definition_sha256": digest(reference)}
    analyzer, adaptations = aggregate.original_analyzer(plan)
    tables = analyzer(component, pair_map, census, support, expected_pairs=len(component_pairs))
    require(len(tables) == 10, "Unexpected original R6 table population")
    output.mkdir(parents=True)
    names = []
    for name, frame in (("S06_repeat_metrics.csv", repeat), ("S07_learner_metrics.csv", learners),
                        ("S08_primary_repeat_ratios.csv", ratios),
                        ("S08_representation_repeat_comparisons.csv", comparisons)):
        frame.to_csv(output / name, index=False, lineterminator="\n")
        names.append(name)
    intermediate = output / "r6_before_companion_filter"
    intermediate.mkdir()
    for name, table in tables.items():
        # aggregate_full_r6:227-230 adds scenario and writes the R6 table.
        # Historical halves are unavailable and are not needed for this filter.
        upstream = intermediate / (name + ".csv")
        table.assign(scenario="corrected_shadow").to_csv(upstream, index=False, lineterminator="\n")
        # Preserve build_r3_tables:278-281 exactly: default read, filter, write.
        current = pd.read_csv(upstream)
        current = current.loc[current.scenario.eq("corrected_shadow")]
        filename = "S29_" + name + ".csv"
        current.to_csv(output / filename, index=False, lineterminator="\n")
        names.append(filename)
    bound_paths = [args.inputs.resolve(), *files.values(), dependency_path, census_path, support_path,
                   internal_path, component_path, Path(__file__).resolve(),
                   HERE / "summarize_r5.py", HERE / "aggregate_full_r6.py", reference]
    if args.pair_strata is not None:
        bound_paths.append(args.pair_strata.resolve())
    receipt = {"status": "MAIN_DATA_FIXED_PREDICTION_REPLAY_COMPLETE", "tables": names,
               "records": len(records), "pairs": len(pairs), "identities": records.identity_hash.nunique(),
               "internal_prediction_rows": len(internal), "component_prediction_rows": len(component),
               "fits": 0, "model_loads": 0, "new_predictions": 0, "bootstrap_calls": 0,
               "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
               "strata_source": str(args.pair_strata.resolve()) if args.pair_strata else str(files["pairs"]),
               "parsing": {"S06_predictions": "round_trip", "S06_canonical_pairs": "pandas default",
                           "S29_predictions_pairs_census_support": "round_trip",
                           "S29_final_companion_filter": "pandas default; read/filter/write"},
               "analyzer_adaptations": [{"old": old, "new": new} for old, new in adaptations],
               "input_bindings": {str(path): digest(path) for path in bound_paths},
               "scope": "14 tables only; no bootstrap primary estimates, historical comparisons, external cases, or scientific gate decision"}
    (output / "REPLAY_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": receipt["status"], "tables": len(names), "output": str(output),
                      "fits": 0, "new_predictions": 0, "bootstrap_calls": 0}))


if __name__ == "__main__":
    main()
