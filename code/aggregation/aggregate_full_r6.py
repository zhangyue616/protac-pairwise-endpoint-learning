"""One full corrected R6 readout, only after all 1,032 slots are complete.

Reuse saved corrected predictions and accepted historical tables. The original
finite-population analyzer is executed with only its pair-count interface
parameterized; no statistical definition, estimator, fit or prediction changes.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_corrected_metadata import prepare

OUT = Path(__file__).resolve().parent
BINDINGS = {}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_bytes(path):
    path = Path(path)
    data = path.read_bytes()
    BINDINGS[path.resolve().as_posix()] = hashlib.sha256(data).hexdigest().upper()
    return data


def read_json(path):
    return json.loads(read_bytes(path))


def read_csv(path):
    return pd.read_csv(io.BytesIO(read_bytes(path)), float_precision="round_trip")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def read_bound(path, expected_sha):
    data = read_bytes(path)
    require(BINDINGS[Path(path).resolve().as_posix()] == expected_sha,
            "Frozen reference changed: " + str(path))
    return data


def prediction_sources(plan):
    sources = plan["ridge_prediction_sources"] + plan["previous_tree_prediction_sources"]
    remaining = plan["remaining_prediction_sources"]
    require(len(remaining) == 80 and len({(x["outer"], x["learner"]) for x in remaining}) == 80,
            "Frozen producer paths for all 80 remaining scopes are not bound")
    sources = sources + remaining
    require(len(sources) == 126 and len({(x["outer"], x["learner"]) for x in sources}) == 126,
            "Expected 126 unique fitted learner/component scopes")
    return sources


def readiness(plan):
    missing = []
    if not plan.get("producer_queue_sha256") or len(plan["remaining_prediction_sources"]) != 80:
        missing.append({"kind": "producer_interface", "detail": "Frozen 80-scope producer index not bound"})
    for row in (plan["ridge_prediction_sources"] + plan["previous_tree_prediction_sources"]
                + plan["remaining_prediction_sources"]):
        for field in ("path", "result"):
            if not Path(row[field]).is_file():
                missing.append({"kind": field, "outer": row["outer"],
                                "learner": row["learner"], "path": row[field]})
        if Path(row["result"]).is_file():
            receipt = json.loads(Path(row["result"]).read_bytes())
            if not str(receipt.get("status", "")).startswith("COMPLETE_BOUNDED_"):
                missing.append({"kind": "incomplete_receipt", "outer": row["outer"],
                                "learner": row["learner"], "path": row["result"],
                                "status": receipt.get("status")})
    return missing


def authority(root, expected_pairs):
    root = Path(root)
    records = read_csv(root / "records_revised.csv").set_index("record_id")
    pairs = read_csv(root / "pairs_revised.csv")
    mapping = read_csv(root / "record_source_group_map.csv").set_index("record_id")
    require(len(records) == 699 and records.index.is_unique, "Record population drift")
    require(len(pairs) == expected_pairs and pairs.pair_id.is_unique, "Pair population drift")
    groups = pairs.record_id_i.map(mapping.component_id)
    require(groups.equals(pairs.record_id_j.map(mapping.component_id)), "Cross-component pair")
    result = pairs[["pair_id", "record_id_i", "record_id_j"]].copy()
    result["heldout_component_id"] = groups
    result["true_delta"] = (pairs.record_id_j.map(records.pdc50_median)
                            - pairs.record_id_i.map(records.pdc50_median))
    require(result.heldout_component_id.nunique() == 43 and not result.isna().any().any(),
            "Authority component/identity mapping incomplete")
    return result.set_index("pair_id")


def validate_predictions(frame, expected, plan):
    require(set(frame.learner) == set(plan["learners"]), "Learner population drift")
    require(set(frame.representation) == set(plan["representations"]), "Representation drift")
    require(set(frame.pipeline) == set(plan["pipelines"]), "Pipeline drift")
    key = ["representation", "learner", "pipeline", "pair_id"]
    require(not frame.duplicated(key).any() and len(frame) == len(expected) * 24,
            "Duplicate/incomplete full corrected prediction grid")
    numeric = frame[["true_delta", "raw_prediction", "scale_c", "calibrated_prediction"]].to_numpy(float)
    require(np.isfinite(numeric).all(), "Nonfinite corrected predictions")
    require(np.allclose(numeric[:, 1] * numeric[:, 2], numeric[:, 3], rtol=0, atol=1e-12),
            "Corrected calibration multiplication does not close")
    for _, part in frame.groupby(key[:-1], sort=True):
        require(set(part.pair_id) == set(expected.index), "Incomplete corrected strategy population")
        aligned = expected.loc[part.pair_id]
        for column in ("record_id_i", "record_id_j", "heldout_component_id"):
            require(np.array_equal(part[column].to_numpy(), aligned[column].to_numpy()),
                    "Corrected prediction identity/direction/component differs: " + column)
        require(np.allclose(part.true_delta.to_numpy(), aligned.true_delta.to_numpy(), rtol=0, atol=1e-12),
                "Corrected target orientation differs")


def original_analyzer(plan):
    source = read_bound(plan["analysis_definition_reference"], plan["analysis_definition_sha256"])
    code = source.decode("utf-8")
    replacements = [("def analyze(predictions, pair_map, census, support):",
                     "def analyze(predictions, pair_map, census, support, expected_pairs=5788):"),
                    ("require(len(pair_map) == 5788 and pair_map.pair_id.is_unique,",
                     "require(len(pair_map) == expected_pairs and pair_map.pair_id.is_unique,"),
                    ("require(len(predictions) == 5788 * 2 * 3 * 4,",
                     "require(len(predictions) == expected_pairs * 2 * 3 * 4,")]
    for old, new in replacements:
        require(code.count(old) == 1, "Frozen analyzer adaptation MATCH_COUNT != 1")
        code = code.replace(old, new, 1)
    namespace = {"__name__": "r6_frozen_finite_population_analyzer",
                 "__file__": plan["analysis_definition_reference"]}
    exec(compile(code, plan["analysis_definition_reference"], "exec"), namespace)
    return namespace["analyze"], replacements


def compare(historical, corrected, keys, fields):
    result = historical[keys + fields].merge(corrected[keys + fields], on=keys,
                                             suffixes=("_historical", "_corrected_shadow"),
                                             validate="one_to_one")
    for field in fields:
        result[field + "_corrected_minus_historical"] = (
            result[field + "_corrected_shadow"] - result[field + "_historical"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Compute only once after complete inputs")
    args = parser.parse_args()
    require(not (OUT / "SUMMARY.json").exists(), "Completed full R6 summary is frozen")
    plan = read_json(OUT / "SUMMARY_PLAN.json")
    missing = readiness(plan)
    if not args.execute or missing:
        status = {"status": "READY_FOR_ONE_FULL_R6_SUMMARY" if not missing else "WAITING_FOR_COMPLETE_R6_INPUTS",
                  "missing": missing, "outcome_values_computed": False,
                  "new_fits": 0, "estimator_loads": 0, "new_predictions": 0}
        write_json(OUT / "INPUT_READINESS.json", status)
        print(json.dumps({"status": status["status"], "missing_items": len(missing),
                          "outcome_values_computed": False}))
        if missing and args.execute:
            raise RuntimeError("Full R6 inputs incomplete; no aggregate result was produced")
        return
    for item in plan["frozen_reference_bindings"]:
        read_bound(item["path"], item["sha256"])
    read_bound(plan["producer_queue"], plan["producer_queue_sha256"])
    authorization = read_json(plan["user_authorization"])
    require(authorization["status"] == "USER_OPTION_B_AUTHORIZED_FULL_REMAINING_R6_ONLY",
            "Full remaining R6 authorization changed")
    helper = read_json(plan["dep006_helper_compatibility"])
    require(helper["status"] == "PASS_DEP006_CURRENT_HELPER_COMPATIBILITY_FROM_ACCEPTED_EVIDENCE",
            "DEP006 current-helper compatibility unresolved")
    sources = prediction_sources(plan)
    frames, fit_units = [], 0
    for row in sources:
        result = read_json(row["result"])
        counts = result["actual_accumulated_fit_ledger_counts"]
        expected_fits = 136 if row["learner"] == "ridge" else 272
        require(str(result["status"]).startswith("COMPLETE_BOUNDED_")
                and counts.get("FIT_ATTEMPT", 0) == expected_fits
                and counts.get("FIT_COMPLETE", 0) == expected_fits
                and counts.get("FIT_FAILURE", 0) == 0, "Incomplete fitted scope receipt")
        frame = read_csv(row["path"])
        require(set(frame.heldout_component_id) == {row["outer"]}
                and set(frame.learner) == {row["learner"]}, "Wrong fitted scope source")
        frames.append(frame)
        fit_units += expected_fits
    dep006 = read_csv(plan["dep006_zero_fit_predictions"])
    dep006 = dep006.loc[dep006.learner.isin(plan["learners"])].copy()
    dep006["heldout_component_id"] = "DEP_006"
    frames.append(dep006)
    corrected = pd.concat(frames, ignore_index=True)
    expected = authority(plan["shadow_input_root"], plan["expected_new_pairs"])
    validate_predictions(corrected, expected, plan)
    require(fit_units == 28560, "Full corrected fit-unit sum does not close")
    historical = {}
    for item in plan["historical_tables"]:
        table_bytes = read_bound(item["path"], item["sha256"])
        historical[item["name"]] = pd.read_csv(io.BytesIO(table_bytes), float_precision="round_trip")
    analyzer, adaptations = original_analyzer(plan)
    output = OUT / "results"
    require(not output.exists(), "Use a fresh full R6 result output directory")
    metadata = OUT / "corrected_metadata"
    if metadata.exists():
        metadata_binding = read_json(metadata / "METADATA_BINDING.json")
        require(metadata_binding["status"] == "CORRECTED_METADATA_PREPARED_USING_ORIGINAL_DEFINITIONS",
                "Corrected metadata preparation is incomplete")
        for path, expected_sha in metadata_binding["input_bindings"].items():
            read_bound(path, expected_sha)
        pair_map = read_csv(metadata / "pair_source_group_map.csv")
        census = read_csv(metadata / "group_census.csv")
        support = read_csv(metadata / "group_support.csv")
    else:
        pair_map, census, support = prepare(plan["shadow_input_root"], metadata,
                                           plan, read_csv, read_bytes, BINDINGS)
    corrected_tables = analyzer(corrected, pair_map, census, support,
                                expected_pairs=plan["expected_new_pairs"])
    output.mkdir(parents=True)
    for name, table in corrected_tables.items():
        require(name in historical, "Historical table counterpart missing: " + name)
        combined = pd.concat([historical[name].assign(scenario="historical"),
                              table.assign(scenario="corrected_shadow")], ignore_index=True)
        combined.to_csv(output / (name + ".csv"), index=False, lineterminator="\n")
    comparisons = {
        "PRIMARY_HISTORICAL_COMPARISON": compare(
            historical["PRIMARY_SUMMARY"], corrected_tables["PRIMARY_SUMMARY"], ["representation", "scope"],
            ["theta_E1_minus_E0matched", "spearman_support_delta", "groups_E1_better",
             "groups_E0matched_better", "groups_tied"]),
        "LEARNER_HISTORICAL_COMPARISON": compare(
            historical["LEARNER_CONTRAST_SUMMARY"], corrected_tables["LEARNER_CONTRAST_SUMMARY"],
            ["representation", "learner"], ["theta_E1_minus_E0matched"]),
        "MACRO_HISTORICAL_COMPARISON": compare(
            historical["MACRO_METRICS"], corrected_tables["MACRO_METRICS"],
            ["representation", "learner", "pipeline"],
            ["group_macro_mae", "group_macro_rmse", "group_macro_mae_minus_zero",
             "raw_group_macro_mae", "raw_group_macro_rmse"]),
        "POOLED_HISTORICAL_COMPARISON": compare(
            historical["POOLED_METRICS"], corrected_tables["POOLED_METRICS"],
            ["representation", "learner", "pipeline"],
            ["pair_pooled_mae", "pair_pooled_rmse", "raw_pair_pooled_mae", "raw_pair_pooled_rmse"])}
    for name, table in comparisons.items():
        table.to_csv(output / (name + ".csv"), index=False, lineterminator="\n")
    corrected.to_csv(output / "all_corrected_source_predictions.csv", index=False, lineterminator="\n")
    read_bytes(Path(__file__))
    read_bytes(OUT / "prepare_corrected_metadata.py")
    output_bindings = {}
    for path in sorted(list(output.rglob("*")) + list(metadata.rglob("*"))):
        if path.is_file():
            output_bindings[path.relative_to(OUT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    # Dataframe JSON converts undefined rank correlations to null.
    primary = json.loads(corrected_tables["PRIMARY_SUMMARY"].to_json(orient="records"))
    historical_primary = json.loads(historical["PRIMARY_SUMMARY"].to_json(orient="records"))
    write_json(OUT / "SUMMARY.json", {
        "status": "COMPLETE_FINITE_POPULATION_43_COMPONENT_THREE_LEARNER_R6_SUMMARY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "components": 43, "learners": plan["learners"], "final_slots": 1032,
        "corrected_pair_rows_per_strategy": plan["expected_new_pairs"],
        "historical_pair_rows_per_strategy": plan["expected_old_pairs"],
        "corrected_prediction_rows": len(corrected), "corrected_fit_units_from_receipts": fit_units,
        "new_fits_this_analysis": 0, "estimator_loads": 0, "new_predictions_this_analysis": 0,
        "primary_corrected": primary, "primary_historical_reused": historical_primary,
        "historical_tables_recomputed": False,
        "confidence_intervals_or_pvalues_computed": False,
        "scientific_definition_changed": False,
        "analyzer_in_memory_interface_adaptations": [{"old": old, "new": new} for old, new in adaptations],
        "interpretation": "Fixed source/shared-identity components and overlapping training; finite-population descriptive comparison. No family-independent validation, total-training uncertainty or validated model-choice rule.",
        "comparison_limit": "Joint repaired graphs/training/OOF calibration/selection derivative; corrected5804 vs historical5788 pair populations. Not an isolated causal graph effect or new formal freeze approval.",
        "scientific_source_author_release_gates_closed": False,
        "independent_stage_acceptance": "PENDING_SINGLE_FINAL_STAGE_ACCEPTANCE",
        "input_bindings": BINDINGS, "output_bindings": output_bindings})
    print(json.dumps({"status": "COMPLETE_FINITE_POPULATION_R6_SUMMARY", "final_slots": 1032,
                      "additional_fits": 0, "additional_predictions": 0}))


if __name__ == "__main__":
    main()
