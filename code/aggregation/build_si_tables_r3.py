"""Format the 22 existing SI tables from completed, frozen R5 outputs.

No fitting, prediction, selection, bootstrap, source admission, or rendering.
The owner must explicitly supply --all-results-ready after summary/diagnostics.
Only this script's responsibility directory is writable. Outputs are table
blocks, their CSVs, local companion copies, and a provenance/index receipt.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import shutil
import sys

sys.dont_write_bytecode = True

LEARNERS = {"ridge": "Ridge", "random_forest": "RF", "xgboost": "XGB"}
ARMS = {"direct_pair": "Direct", "endpoint_E0": "E0", "endpoint_E1": "E1",
        "endpoint_E0_matched": "E0-matched", "context_pair_C1": "C1", "zero": "Zero"}
REPRESENTATIONS = {"binary": "Binary", "count": "Count"}
GROUPS = ["persistent_zero", "restored", "both_nonzero"]
STATISTICS = {
    "gamma_direct_pair": "γ Direct", "gamma_endpoint_E0": "γ E0",
    "gamma_endpoint_E1": "γ E1", "gamma_endpoint_E0_matched": "γ E0-matched",
    "gamma_context_pair_C1": "γ C1 (trees)", "theta_binary": "θ Binary",
    "theta_count": "θ Count", "delta": "δ Count−Binary",
    "theta_match_binary": "θ match Binary", "theta_match_count": "θ match Count",
    "theta_support_binary": "θ support Binary", "theta_support_count": "θ support Count",
    "log_C1_over_direct_binary": "log(C1/Direct), Binary",
    "log_C1_over_direct_count": "log(C1/Direct), Count",
}

# These schemas are read only after the explicit completion gate. They also
# expose the exact consumer interface for the owner's static integration check.
SUMMARY_SCHEMAS = {
    "edge_coverage.csv": ["pair_id", "coverage_repeats"],
    "fold_counts.csv": ["repeat", "outer_fold", "test_edges"],
    "repeat_metrics.csv": ["representation", "learner", "pipeline", "repeat", "stratum", "scale", "n", "n_direction", "mae", "rmse", "DA_strict", "DA_half", "zero_predictions", "true_ties_excluded"],
    "learner_metrics.csv": ["representation", "learner", "pipeline", "stratum", "scale", "observations", "repeats_nonempty", "mean_mae", "min_mae", "max_mae", "mean_DA_strict", "mean_DA_half"],
    "primary_estimates.csv": ["statistic", "estimate", "lower_95", "upper_95", "draws_requested", "valid_draws", "invalid_draws", "component_count", "seed"],
    "selection_boundary_summary.csv": ["representation", "learner", "pipeline", "parameter", "outer_models", "at_min", "at_max"],
    "source_metrics.csv": ["holdout_id", "source_index", "learner", "pipeline", "scenario", "scale", "n", "mae"],
    "external_metrics.csv": ["representation", "learner", "pipeline", "scale", "n", "n_direction", "mae", "rmse", "DA_strict", "DA_half", "zero_predictions"],
}
DIAGNOSTIC_SCHEMAS = {
    "main_consistency_learner_mean.csv": ["representation", "learner", "pipeline", "calibrated_kappa", "calibrated_mae", "calibrated_projected_mae", "calibrated_projected_rmse"],
    "external_consistency.csv": ["representation", "learner", "pipeline", "calibrated_kappa", "calibrated_mae", "calibrated_projected_mae", "calibrated_projected_rmse"],
    "E1_final_iteration5_summary.csv": ["representation", "learner", "final_models", "training_h_change_rmse_median", "training_h_change_rmse_min", "training_h_change_rmse_max", "alpha_change_record_weighted_rmse_median", "alpha_change_record_weighted_rmse_min", "alpha_change_record_weighted_rmse_max"],
    "raw_representation_classes.csv": ["population_representation", "records", "chemical_identities", "representation_classes", "classes_with_distinct_chemical_identities", "records_in_cross_identity_classes"],
    "effective_input_scope_summary.csv": ["population", "scope", "repeat", "outer_fold", "records", "vocabulary_size", "effective_representation_classes", "classes_with_distinct_chemical_identities", "effective_zero_vectors", "unknown_count_fraction_mean", "unknown_count_fraction_max"],
    "outer_effective_forced_zero_bounds.csv": ["representation", "repeat", "outer_fold", "test_pairs", "test_zero_difference_pairs", "test_forced_zero_mae_contribution"],
    "posthoc_internal_group_census.csv": ["repeat", "outer_fold", "group", "observations", "unique_edges", "chemical_identities", "records", "condition_blocks", "accepted_internal_dependency_components"],
    "posthoc_internal_calibrated_loss_contributions.csv": ["learner", "pipeline", "persistent_zero", "restored", "both_nonzero", "overall_count_minus_binary_mae"],
    "pair_input_class_census.csv": ["representation", "population", "pipeline", "repeat", "outer_fold", "pairs", "canonical_input_classes", "exact_zero_difference_pairs"],
    "external_member_prediction_diagnostics.csv": ["representation", "learner", "pipeline", "repeat", "outer_fold", "external_pairs", "calibrated_zero_predictions"],
    "external_ensemble_prediction_diagnostics.csv": ["representation", "learner", "pipeline", "external_pairs", "calibrated_zero_predictions", "calibrated_distinct_prediction_classes_tol"],
    "posthoc_external_calibrated_loss_contributions.csv": ["learner", "pipeline", "persistent_zero", "restored", "both_nonzero", "overall_count_minus_binary_mae"],
    "posthoc_external_ensemble_group_census.csv": ["group", "observations", "chemical_identities", "records", "condition_blocks", "external_series_components"],
}
COMPANIONS = {
    2: [("bundle", "input_build/config/layer_support_census.csv", "All internal and source outer/inner training support; matched masks")],
    5: [("description", "graph_block_completeness.csv", "Every revised condition block and compatible graph completeness"), ("bundle", "input_build/config/layer_record_support.csv", "Layer-specific E1 and E0-matched shared support")],
    6: [("summary", "repeat_metrics.csv", "All repetition/stratum/raw-calibrated metrics and Zero"), ("summary", "outer_predictions_long.csv", "Complete current outer edge predictions")],
    7: [("summary", "learner_metrics.csv", "All repetition-aggregated metrics and support, including raw and Zero")],
    8: [("summary", "primary_estimates.csv", "All contrasts with actual valid/invalid conditional draws"), ("summary", "primary_repeat_ratios.csv", "Per-repeat strategy ratios and matched/support absolute differences"), ("summary", "representation_repeat_comparisons.csv", "Per-repeat representation ratios and absolute differences")],
    9: [("summary", "selected_candidates.csv", "Selected candidates and frozen inner scales"), ("summary", "selection_parameter_boundaries.csv", "Full selected parameters and tested numerical endpoints")],
    10: [("diagnostics", "main_consistency_learner_mean.csv", "Full internal projection/error metrics"), ("diagnostics", "main_consistency_repeat.csv", "Repeat-level consistency before equal-repeat averaging"), ("diagnostics", "external_consistency.csv", "Full exploratory ensemble projection/error metrics")],
    11: [("diagnostics", "E1_final_training_iteration_stability.csv", "Five rounds for every selected final tree E1 model"), ("diagnostics", "E1_final_iteration5_summary.csv", "Actual median/min/max fourth-to-fifth changes")],
    12: [("diagnostics", "effective_input_scope_summary.csv", "Every actual vocabulary input scope and unknown fraction"), ("diagnostics", "outer_vocabulary_census.csv", "Actual vocabulary/record/input-class support"), ("diagnostics", "raw_main_pair_diagnostics.csv", "Raw full-graph pair equality, distinct from actual fold constraints")],
    13: [("diagnostics", "posthoc_internal_group_census.csv", "Pooled, repeat and fold input-defined group support")],
    14: [("diagnostics", "posthoc_internal_calibrated_three_group_details.csv", "All current calibrated model/representation/group metrics and own-group Zero"), ("diagnostics", "posthoc_internal_group_repeatmean_range.csv", "Raw/calibrated group metrics and repeat sensitivity ranges"), ("diagnostics", "posthoc_internal_actual_structural_zero_contribution.csv", "Actual per-repeat denominator and structural-zero contributions")],
    15: [("bundle", "input_build/config/selected_sources.json", "Retained S0/S1/S3 index, seed and support; S2 retired"), ("bundle", "input_build/config/source_inner_identity_folds.csv", "Retained, filtered source inner identity assignments")],
    16: [("summary", "source_metrics.csv", "Every current source/scenario/arm/raw-calibrated metric and Zero"), ("summary", "source_predictions.csv", "All current edge-level source-held-out predictions"), ("summary", "source_selected_candidates.csv", "Source candidate and calibration selections")],
    18: [("description", "revised_entity_source_comparisons.csv", "Corrected source-row recurrence comparisons"), ("description", "repeated_pair_source_comparisons.csv", "Repaired-graph repeated-pair comparisons including two additions"), ("description", "DATA_DESCRIPTION.json", "Current population and repaired-graph recurrence evidence")],
    19: [("bundle", "old_exploratory_case/records.csv", "Bound fixed seven compounds and unchanged labels"), ("description", "external_nearest_training_similarity.csv", "Current-699 structural neighborhood only; no prediction or source admission")],
    20: [("summary", "external_metrics.csv", "All current exploratory raw/calibrated ensemble metrics and Zero"), ("summary", "external_ensemble_predictions.csv", "Current 21-edge ensembles"), ("summary", "external_per_model_predictions.csv", "All 15 current members for each model combination")],
    21: [("diagnostics", "external_member_prediction_diagnostics.csv", "Every current member output class/zero diagnostic"), ("diagnostics", "external_ensemble_prediction_diagnostics.csv", "Every current ensemble output class/zero diagnostic"), ("diagnostics", "pair_input_class_census.csv", "All actual Direct d and C1 [d,s] class counts")],
    22: [("diagnostics", "posthoc_external_calibrated_three_group_details.csv", "All current exploratory ensemble/representation/group metrics and own-group Zero"), ("diagnostics", "posthoc_external_group_member_repeatmean_range.csv", "Member loss kept separate from ensemble loss"), ("diagnostics", "posthoc_external_ensemble_zero_member_evidence.csv", "Current member/ensemble zero-origin evidence")],
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def number(value):
    if value is None or str(value).strip().lower() in ("", "na", "nan", "none", "null"):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def n(value):
    result = number(value)
    return "NA" if result is None else str(int(result))


def fmt(value, places=6):
    result = number(value)
    if result is None:
        return "NA"
    return f"{result:.3e}" if result != 0 and abs(result) < 10 ** -places else f"{result:.{places}f}"


def sorted_model(rows):
    return sorted(rows, key=lambda r: (list(REPRESENTATIONS).index(r.get("representation", "binary")),
                  list(LEARNERS).index(r["learner"]), list(ARMS).index(r["pipeline"]),
                  int(r.get("repeat", 0))))


def model_key(row):
    return tuple(row[field] for field in ("representation", "learner", "pipeline"))


def model_labels(row):
    return [REPRESENTATIONS[row["representation"]], LEARNERS[row["learner"]], ARMS[row["pipeline"]]]


def one(rows, **keys):
    matches = [row for row in rows if all(str(row[k]) == str(v) for k, v in keys.items())]
    require(len(matches) == 1, f"expected one row for {keys}, got {len(matches)}")
    return matches[0]


def value_range(values):
    values = [int(float(x)) for x in values]
    require(bool(values), "empty range support")
    lo, hi = min(values), max(values)
    return str(lo) if lo == hi else f"{lo}–{hi}"


def local(root, relative):
    value = Path(relative)
    require(not value.is_absolute() and not value.drive, "expected bundle-relative path")
    path = (root / value).resolve()
    require(path.is_relative_to(root), "input path escapes root")
    return path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest().upper()


class Builder:
    def __init__(self, roots, output):
        self.roots, self.output = roots, output
        self.sources, self.panels, self.sections, self.companions = {}, [], [], []

    def track(self, path):
        path = path.resolve()
        require(path.is_file(), f"missing final dependency: {path}")
        if path.as_posix() not in self.sources:
            self.sources[path.as_posix()] = {"path": path.as_posix(), "sha256": digest(path)}
        return path

    def json(self, path):
        return json.loads(self.track(path).read_text(encoding="utf-8-sig"))

    def read(self, path, columns=()):
        with self.track(path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            require(set(columns) <= set(reader.fieldnames or []), f"schema mismatch: {path}; required {columns}")
            return list(reader)

    def panel(self, sid, label, headers, rows, evidence):
        require(all(len(row) == len(headers) for row in rows), f"S{sid}{label} width mismatch")
        filename = f"tables/S{sid:02d}{label}.csv"
        self.panels.append({"table": f"S{sid}", "panel": label, "file": filename,
                            "columns": headers, "rows": rows,
                            "evidence": [self.track(path).as_posix() for path in evidence]})
        escape = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
        return "| " + " | ".join(map(escape, headers)) + " |\n| " + " | ".join("---" for _ in headers) + " |\n" + "".join(
            "| " + " | ".join(map(escape, row)) + " |\n" for row in rows)

    def table(self, sid, title, caption, panels):
        self.sections.append((sid, f"### Table S{sid}. {title}\n\n{caption}\n\n" + "\n\n".join(panels)))

    def finish(self, spec, summary, diagnostics):
        require([sid for sid, _ in self.sections] == list(range(1, 23)), "not exactly S1–S22 in original order")
        for sid, entries in COMPANIONS.items():
            for root, relative, scope in entries:
                source = self.track(local(self.roots[root], relative))
                target = f"companions/S{sid:02d}_{source.name}"
                self.companions.append({"table": f"S{sid}", "file": target, "source": source.as_posix(),
                    "sha256": self.sources[source.as_posix()]["sha256"], "scope": scope,
                    "public_status": "local review only; no current public release established"})
        # All scientific dependencies and table closure are read/checked before
        # creating the output directory. Existing output is never overwritten.
        require(not self.output.exists(), "output exists; use a new directory under tables_source_build")
        self.output.mkdir(parents=True)
        (self.output / "tables").mkdir()
        (self.output / "companions").mkdir()
        (self.output / "TABLES.md").write_text("\n\n".join(text for _, text in self.sections) + "\n", encoding="utf-8", newline="\n")
        panel_manifest = []
        for panel in self.panels:
            with (self.output / panel["file"]).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(panel["columns"])
                writer.writerows(panel["rows"])
            panel_manifest.append({**panel, "rows": len(panel["rows"])})
        for row in self.companions:
            shutil.copyfile(row["source"], self.output / row["file"])
        with (self.output / "COMPANION_INDEX.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["table", "file", "source", "sha256", "scope", "public_status"], lineterminator="\n")
            writer.writeheader()
            writer.writerows(self.companions)
        report = {"status": "COMPLETE_R5_SI_TABLE_BLOCKS_NOT_MANUSCRIPT_ACCEPTANCE", "created_utc": datetime.now(timezone.utc).isoformat(),
                  "run_id": spec["run_id"], "numbered_tables": 22, "panels": panel_manifest, "companions": self.companions,
                  "sources": list(self.sources.values()), "scientific_fits_added": 0, "new_predictions": 0,
                  "new_bootstrap_draws_tests_or_source_admissions": 0, "formatting_only": True,
                  "summary_status": summary["status"], "diagnostics_status": diagnostics["status"],
                  "actual_prediction_components": summary["actual_prediction_components"],
                  "S11": "median [minimum, maximum] of actual selected final tree E1 fifth-round diagnostics",
                  "S12_unknown_count_fraction": "per-record mean and maximum; not median",
                  "S18": "accepted corrected graph/source-row transport, current condition/source recurrence",
                  "limitations": ["No full SI texts, Word/PDF or rendering produced", "Conditional intervals remain fixed-prediction only", "Seven-compound case remains exploratory; new-source admission is zero", "Source holdouts are S0/S1/S3, binary only, original arms only", "Owner holds one final consolidated numerical/scientific/visual acceptance"],
                  "outputs": [{"file": p.relative_to(self.output).as_posix(), "sha256": digest(p)} for p in sorted(self.output.rglob("*")) if p.is_file()]}
        (self.output / "PROVENANCE.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        return {"status": report["status"], "numbered_tables": 22, "panels": len(panel_manifest), "companions": len(self.companions), "output": self.output.as_posix()}


def prior_table_rows(text, sid):
    match = re.search(rf"^### Table S{sid}\. .*?(?=^### Table S\d+\.|^## Supplementary Text|\Z)", text, re.M | re.S)
    require(match is not None, f"R4 table S{sid} absent")
    return [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in match.group().splitlines()
            if line.startswith("| ") and not line.startswith("| ---")][1:]


def build(args):
    responsibility = Path(__file__).resolve().parent
    output = args.output.resolve() if args.output else responsibility / "generated"
    require(output != responsibility and output.is_relative_to(responsibility), "output must be a new child of tables_source_build")
    require(args.all_results_ready, "owner must authorize --all-results-ready after completed summary and diagnostics")
    roots = {"bundle": args.bundle.resolve(), "summary": args.summary.resolve(),
             "diagnostics": args.diagnostics.resolve(), "description": args.data_description.resolve()}
    builder = Builder(roots, output)
    bundle, sp, dp, ddp = (roots[k] for k in ("bundle", "summary", "diagnostics", "description"))
    spec = builder.json(bundle / "spec.json")
    require(spec["schema_id"] == "R5_PORTABLE_TRAINING_V1", "unexpected model specification")
    acceptance = builder.json(STAGE_ONE / "FINAL_ACCEPTANCE.json")
    require(acceptance["status"].startswith("PASS_SINGLE_CONCENTRATED_STAGE1"), "current results not accepted")
    contract = builder.json(CURRENT_R5 / "EXECUTION_CONTRACT.json")
    spec["run_id"] = contract["run_id"]
    summary = builder.json(sp / "SUMMARY.json")
    diag = builder.json(dp / "DIAGNOSTICS_RECEIPT.json")
    require(summary["run_id"] == diag["run_id"] == spec["run_id"], "accepted analysis run mismatch")
    require(summary["logical_estimator_fit_coverage"] == 17925 and summary["new_fit_calls"] == 17425, "fit accounting drift")
    require(summary["new_external_admitted_sources"] == 0 and not summary["new_external_evaluation_performed"], "unplanned external result")
    cfg = builder.json(local(bundle, spec["scientific_config"]))
    prep = builder.json(local(bundle, spec["input_preparation_receipt"]))
    desc = builder.json(ddp / "DATA_DESCRIPTION.json")
    records = builder.read(local(bundle, spec["inputs"]["records"]), ["record_id", "identity_hash", "normalized_poi_id", "pair_context_hash"])
    external = builder.read(local(bundle, spec["external"]["records"]))
    source = builder.json(local(bundle, spec["source_holdout"]["selected_sources"]))["source_holdouts"]
    require([r["source_index"] for r in source] == [0, 1, 3], "source indices changed")
    require(not spec["source_holdout"]["new_count_holdout"] and not spec["source_holdout"]["matched_arm_in_source_holdout"], "source scope expanded")
    census_path = bundle / "input_build/config/layer_support_census.csv"
    census = builder.read(census_path, ["scope", "layer", "repeat", "outer_fold", "inner_fold", "training_records", "training_identities", "e1_training_records", "e0matched_training_records", "train_pairs", "validation_pairs", "vocabulary_size"])
    require(all(r["e1_training_records"] == r["e0matched_training_records"] for r in census), "matched support differs")
    require(len(records) == desc["records"] == diag["main_records"] == 699, "record population differs")
    require(desc["pairs"] == diag["main_pairs"] == 5804, "repaired graph population differs")
    require(len(external) == 7, "fixed BRD4 case changed")
    jobs = spec["jobs"]
    combinations = {tuple(j[k] for k in ("representation", "learner", "pipeline")) for j in jobs}
    require(len(jobs) == 15 * len(combinations) == summary["internal_models"] == 420, "model population differs")
    prior_path = builder.track(args.prior_si.resolve())
    prior = prior_path.read_text(encoding="utf-8-sig")
    sm = {name: builder.read(sp / name, columns) for name, columns in SUMMARY_SCHEMAS.items()}
    dm = {name: builder.read(dp / name, columns) for name, columns in DIAGNOSTIC_SCHEMAS.items()}
    S = lambda name: sp / name
    D = lambda name: dp / name
    panel, table = builder.panel, builder.table

    old_s1 = prior_table_rows(prior, 1)
    historical = [row for row in old_s1 if row[0] in {"Frozen TACK snapshot", "Operator-rule exclusions", "Retained endpoint / aggregation layer", "Historical POI/label-eligible pool", "Excluded coarse-category labels", "Historical graph edges removed"}]
    require(len(historical) == 6, "historical construction rows changed; owner must review")
    excluded = builder.json(bundle / "input_build/source/exclusions_24.json")
    reasons = Counter(r["reason_group"] for r in excluded)
    require(len(excluded) == 24 and len({r["record_id"] for r in excluded}) == 24, "historical exclusion scope differs")
    fixed = sum(desc["record_time_status"].get(k, 0) for k in ("stored_numeric", "source_method_fixed_time", "corrected_source_fixed_time"))
    unknown = sum(v for k, v in desc["record_time_status"].items() if k.startswith("unknown"))
    rows = historical + [
        ["Previously accepted continuous pool", str(len(builder.read(bundle / "input_build/source/records_723.csv"))) + " records", "Historical starting pool before the fixed R5 exclusions"],
        ["R5 record exclusions", str(len(excluded)), "; ".join(f"{key}: {count}" for key, count in sorted(reasons.items())) + "; unclosed source binding is not a confirmed numerical error"],
        ["Current continuous pool", f"{desc['records']} records / {desc['identities']} identities", "Retained labels unchanged; all known range midpoints excluded"],
        ["Current recorded-condition blocks", str(desc["blocks"]), f"{desc['singleton_blocks']} singleton; {desc['nonsingleton_blocks']} with ≥2 records"],
        ["Compatible graph", f"{desc['pairs']} edges / {desc['endpoint_records']} endpoint records", "Distinct identities; retained binary Tanimoto ≥0.5 eligibility"],
        ["Eligible records outside graph endpoints", str(desc["eligible_records_outside_graph"]), "May contribute to endpoint training"],
        ["Historical 24-record exclusion", "191 edges removed; 5979 to 5788", "Deletion-only step before the present structure repair"],
        ["Source-depicted graph repairs", "7 identities / 12 records", "One-carbon bridge corrections; Example 17 also adds the source-drawn N-methyl; labels unchanged"],
        ["Graph update after structure repair", "3 removed / 19 added; net +16", "5788 to 5804 compatible edges under unchanged eligibility rules"],
        ["Cliffs", str(desc["cliffs"]), "Secondary |y|≥1 stratum with retained numerical tolerance"],
        ["Fixed-time / shared-window / unknown-time records", f"{fixed} / {desc['record_time_status']['source_method_common_window']} / {unknown}", "Fixed-time provenance remains stored/source-method/corrected-source; 16–18 h retained as a window"],
        ["Fixed-time / shared-window / unknown-time edges", " / ".join(str(desc["pair_time_status"][k]) for k in ("same_recorded_fixed_time", "common_reported_time_window", "recorded_context_time_unknown")), "Same recorded-status within eligible pairs"],
        ["S0 raw3229 DC50", "1.2 nM", "Confirmed original decimal; retained without a label correction"],
    ]
    table(1, "Development-data construction and source curation.", f"Counts describe the retained source-to-record construction and revised continuous analysis. A record is a molecular-identity–POI–recorded-context entry, not an independent assay run. Exposure-time evidence retains its stated limitations. {reasons['known_range_midpoint']} known range midpoints and {len(excluded)-reasons['known_range_midpoint']} records with unclosed source-table binding are excluded under the fixed R5 decision. DC50 is in nM; prediction errors use pDC50.", [panel(1, "a", ["Stage or property", "Count / value", "Definition"], rows, [prior_path, ddp / "DATA_DESCRIPTION.json", bundle / spec["input_preparation_receipt"], bundle / "input_build/source/exclusions_24.json"])])

    outer = sorted([r for r in census if r["scope"] == "internal" and int(r["inner_fold"]) == -1], key=lambda r: (int(r["repeat"]), int(r["outer_fold"])))
    fold_maps = {(int(r["repeat"]), r["identity_hash"]): int(r["outer_fold"]) for r in builder.read(local(bundle, spec["inputs"]["outer_folds"]))}
    rows = []
    for r in outer:
        repeat, fold = int(r["repeat"]), int(r["outer_fold"])
        train = [x for x in records if fold_maps[(repeat, x["identity_hash"])] != fold]
        blocks = Counter((x["normalized_poi_id"], x["pair_context_hash"]) for x in train)
        rows.append([f"R{repeat}/F{fold}", n(r["training_identities"]), str(desc["identities"] - int(r["training_identities"])), n(r["training_records"]), str(len(records) - len(train)), n(r["train_pairs"]), n(r["validation_pairs"]), str(sum(v >= 2 for v in blocks.values())), n(r["e1_training_records"])])
    require(len(rows) == 15, "outer support lacks 3×5 rows")
    totals = [sum(int(r["validation_pairs"]) for r in outer if int(r["repeat"]) == repeat) for repeat in range(3)]
    coverage = Counter(int(r["coverage_repeats"]) for r in sm["edge_coverage.csv"])
    require(sum(coverage.values()) == desc["pairs"] and sum(k*v for k, v in coverage.items()) == summary["test_pair_observations"], "coverage closure differs")
    table(2, "Identity-fold support and actual test-edge coverage.", f"Both endpoint identities must belong to the indicated training or held-out set. Original outer assignments (seeds {', '.join(map(str, cfg['splitting']['outer_split_seeds']))}) and each original outer-cell inner map are filtered without reassignment. Test observations total {'/'.join(map(str, totals))}, covering {summary['unique_test_pairs']} different edges through {summary['test_pair_observations']} observations. Of {desc['pairs']} graph edges, {'/'.join(str(coverage.get(k, 0)) for k in range(4))} are tested in 0/1/2/3 repetitions. Training blocks ≥2 and E1/E0-matched training records use the current layer support; the matched masks are equal.", [panel(2, "a", ["Split", "Train IDs", "Test IDs", "Train rec.", "Test rec.", "Train edges", "Test edges", "Train blocks ≥2", "E1/matched rec."], rows, [census_path, S("edge_coverage.csv"), sp / "SUMMARY.json"])])

    vocab = builder.json(local(bundle, spec["inputs"]["vocab_manifest"]))
    rconf = cfg["representation"]
    rows = [["Encoding", f"Presence; {cfg['data']['fingerprint']['n_bits']} folded positions", "Raw counts; unfolded 32-bit environment IDs"],
            ["Radius / chirality", f"{rconf['radius']} / {rconf['use_chirality']}", f"{rconf['radius']} / {rconf['use_chirality']}"],
            ["Atom invariants", "Standard connectivity", rconf["atom_invariants"]],
            ["Bond types / ring membership", f"{rconf['use_bond_types']} / {rconf['include_ring_membership']}", f"{rconf['use_bond_types']} / {rconf['include_ring_membership']}"],
            ["Only nonzero invariants / redundant environments", "False / false", f"{rconf['only_nonzero_invariants']} / {rconf['include_redundant_environments']}"],
            ["Input transformations", "Retained binary vectors", "No log, standardization, normalization, or rebinarization"],
            ["Vocabulary", "Fixed 2,048 positions", "Sorted IDs from all eligible training-identity records, before label-support filtering"],
            ["Unknown held-out / external IDs", "Not applicable to fixed positions", "Discarded; no vocabulary expansion"],
            ["Actual layer dimensions", "2,048", f"{value_range(x['vocabulary_size'] for x in vocab.values())} across {len(vocab)} layers; {value_range(x['vocabulary_size'] for x in outer)} across {len(outer)} outer layers"],
            ["Raw eligible-training environment-ID union", "Not applicable", str(desc["raw_environment_identifiers"])],
            ["Input storage", "Float-valued binary matrix", "Dense count matrix; no modulo folding"],
            ["Generator / version", "Retained stored vectors", f"RDKit GetSparseCountFingerprint / {cfg['execution']['versions']['rdkit']}"],]
    table(3, "Molecular representation and vocabulary settings.", "Both representations use the same retained complete molecular graph and radius 2. Graph eligibility remains the stored binary Tanimoto ≥0.5; count inputs do not redefine the edge population. Dimensions describe actual training vocabularies. Unfolded environment IDs remain hashed 32-bit identifiers and are not collision-free.", [panel(3, "a", ["Property", "Binary", "Count"], rows, [bundle / spec["scientific_config"], local(bundle, spec["inputs"]["vocab_manifest"]), ddp / "DATA_DESCRIPTION.json"])])

    grid = []
    for i, alpha in enumerate(cfg["learners"]["ridge"]["alphas"]):
        grid.append(["Ridge", f"ridge_{i:02d}", str(alpha), "—", "—", "—"])
    for learner, prefix, fields in [("random_forest", "rf", ["max_depth", "min_samples_leaf", "max_features"]), ("xgboost", "xgb", ["max_depth", "min_child_weight", "reg_lambda"])]:
        for i, values in enumerate(itertools.product(*(cfg["learners"][learner]["grid"][field] for field in fields))):
            ordered = [values[0], values[2], values[1]] if prefix == "rf" else list(values)
            grid.append([LEARNERS[learner], f"{prefix}_{i:02d}", *map(str, ordered), "—"])
    require(len(grid) == 3 * cfg["model_selection"]["candidate_count_each_learner"], "candidate grid size differs")
    fixed = [["Ridge", "L2 squared-loss regression; Direct/E1 no intercept; E0 and E0-matched intercept"],
             ["RF", "; ".join(f"{k}={v}" for k, v in cfg["learners"]["random_forest"]["common"].items())],
             ["XGB", "; ".join(f"{k}={v}" for k, v in cfg["learners"]["xgboost"]["common"].items()) + "; CUDA"],
             ["All", f"Model seeds {'/'.join(map(str, cfg['splitting']['model_seeds_paired_by_repeat']))} paired by repetition; source holdouts use {source[0]['model_seed']}"],
             ["Selection", cfg["model_selection"]["selection_metric"] + "; " + cfg["model_selection"]["tie_break"]]]
    table(4, "Complete candidate grids and fixed learner settings.", "Eight candidates are evaluated independently for each learner–arm–representation outer cell. For Ridge, column 3 is α; for RF, columns 3–5 are max_depth, max_features and min_samples_leaf; for XGB, they are max_depth, min_child_weight and reg_lambda. Tested values and selection budgets are shared across representations; scale and dimensionality differ.", [panel(4, "a", ["Learner", "Candidate ID", "Parameter 1", "Parameter 2", "Parameter 3", "Other candidate parameters"], grid, [bundle / spec["scientific_config"]]), panel(4, "b", ["Learner", "Fixed settings"], fixed, [bundle / spec["scientific_config"]])])

    rows = [["Direct", "Ridge/RF/XGB", "(d,y) and (−d,−y)", "Compatible training edges", "0.5[f(d)−f(−d)]"],
            ["E0", "Ridge/RF/XGB", "X / original z", "All eligible training-identity records", "h(X_j)−h(X_i)"],
            ["E0-matched", "Ridge/RF/XGB", "X / original z; E0 intercept retained", "Exactly E1's current layer record mask (blocks ≥2 records)", "h(X_j)−h(X_i)"],
            ["E1 Ridge", "Ridge", "Block-centered X / z", "Current training blocks with ≥2 records", "X_jᵀw−X_iᵀw"],
            ["E1 trees", "RF/XGB", "X / z−α_block; five fixed backfits", "Current training blocks with ≥2 records", "h⁽⁵⁾(X_j)−h⁽⁵⁾(X_i)"],
            ["C1", "RF/XGB", "([d,s],y) and ([−d,s],−y)", "Compatible training edges", "0.5[f([d,s])−f([−d,s])]"],
            ["Zero", "Baseline", "No fit", "Same evaluation edges", "0"]]
    table(5, "Model arms, training support, and structural loss interpretation.", f"For features X_i, d=X_j−X_i, s=X_j+X_i and target y=z_j−z_i, z=pDC50. Count vocabulary uses all eligible current training records before E1/E0-matched label filtering and is shared across arms. POI/context fields are not predictor inputs. The {desc['nonsingleton_blocks']} nonsingleton full-data blocks include {desc['incomplete_nonsingleton_blocks']} incomplete distinct-identity pair graphs; complete-block centered-loss identities do not characterize every graph. E0-matched changes E0 support, retaining original z and its intercept.", [panel(5, "a", ["Arm", "Learners", "Training input / target", "Support", "Pair prediction"], rows, [bundle / spec["scientific_config"], census_path, ddp / "DATA_DESCRIPTION.json"])])

    rm = sm["repeat_metrics.csv"]
    base = sorted_model([r for r in rm if r["stratum"] == "all" and r["scale"] == "calibrated" and r["pipeline"] != "zero"])
    require(len(base) == len(combinations) * 3, "primary model-repeat population incomplete")
    rows_a, rows_b = [], []
    for r in base:
        raw = one(rm, **{k: r[k] for k in ("representation", "learner", "pipeline", "repeat", "stratum")}, scale="raw")
        labels = model_labels(r) + [f"R{r['repeat']}"]
        rows_a.append(labels + [fmt(raw["mae"]), fmt(r["mae"]), fmt(raw["rmse"]), fmt(r["rmse"])])
        rows_b.append(labels + [fmt(r["DA_strict"], 4), fmt(r["DA_half"], 4), n(r["zero_predictions"]), n(r["n_direction"])])
    zeros = [one(rm, representation="binary", learner="ridge", pipeline="zero", repeat=r, stratum="all", scale="raw") for r in range(3)]
    zero_rows = [[f"R{r['repeat']}", n(r["n"]), fmt(r["mae"]), fmt(r["rmse"]), fmt(r["DA_strict"], 4), fmt(r["DA_half"], 4)] for r in zeros]
    table(6, "Complete primary metrics by model and repetition.", f"Panel A includes all {len(base)} representation–learner–arm–repetition rows, including E0-matched; errors are in pDC50. Each repetition pools five folds with equal edge weights. Panel B reports calibrated strict and half-credit DA, numerical-zero predictions (|q|≤10^−12) and nonzero-truth direction denominators. Positive scales preserve raw signs; c=0 may add zeros. R0/R1/R2 have {'/'.join(n(r['n']) for r in zeros)} observations and {'/'.join(n(r['true_ties_excluded']) for r in zeros)} excluded true ties. The common Zero is listed once in Panel C.", ["**Panel A. Error metrics.**\n\n" + panel(6, "a", ["Repr.", "Learner", "Arm", "Rep.", "MAE raw", "MAE cal", "RMSE raw", "RMSE cal"], rows_a, [S("repeat_metrics.csv")]), "**Panel B. Direction and zeros.**\n\n" + panel(6, "b", ["Repr.", "Learner", "Arm", "Rep.", "DA strict", "DA half", "Zero preds", "Direction n"], rows_b, [S("repeat_metrics.csv")]), "**Panel C. Common Zero.**\n\n" + panel(6, "c", ["Rep.", "Edges", "MAE", "RMSE", "DA strict", "DA half"], zero_rows, [S("repeat_metrics.csv")])])

    lm = sm["learner_metrics.csv"]
    strata = [("abs_delta_ge_1", "|y|≥1"), ("g_seen", "POI seen"), ("g_unseen", "POI unseen"), ("known_fixed_time", "Fixed time"), ("common_time_window", "Shared time window"), ("time_unknown", "Time unknown"), ("has_approximate_label", "Approximate label")]
    parts = []
    for i, (st, label) in enumerate(strata):
        data = sorted_model([r for r in lm if r["stratum"] == st and r["scale"] == "calibrated" and r["pipeline"] != "zero"])
        require(len(data) == len(combinations), "stratum model scope incomplete")
        rows = [model_labels(r) + [n(r["observations"]), n(r["repeats_nonempty"]), f"{fmt(r['mean_mae'])} [{fmt(r['min_mae'])}, {fmt(r['max_mae'])}]", fmt(r["mean_DA_strict"], 4), fmt(r["mean_DA_half"], 4)] for r in data]
        if all(int(r["observations"]) == 0 for r in data):
            rows = [["Both", "All", "All", "0", "0", "NA", "NA", "NA"]]
        parts.append(f"**Panel {chr(65+i)}. {label}.**\n\n" + panel(7, chr(97+i), ["Repr.", "Learner", "Arm", "Obs.", "Reps", "MAE mean [min, max]", "DA strict", "DA half"], rows, [S("learner_metrics.csv")]))
    rows = []
    for st, label in strata:
        r = one(lm, representation="binary", learner="ridge", pipeline="zero", stratum=st, scale="raw")
        rows.append([label, n(r["observations"]), n(r["repeats_nonempty"]), f"{fmt(r['mean_mae'])} [{fmt(r['min_mae'])}, {fmt(r['max_mae'])}]", fmt(r["mean_DA_strict"], 4), fmt(r["mean_DA_half"], 4)])
    parts.append("**Panel H. Common stratum-specific Zero.**\n\n" + panel(7, "h", ["Stratum", "Obs.", "Reps", "MAE mean [min, max]", "DA strict", "DA half"], rows, [S("learner_metrics.csv")]))
    table(7, "All prespecified supplementary evaluation strata.", "Calibrated MAE means and ranges summarize nonempty repetitions; ranges show split sensitivity, not confidence intervals. Obs. counts repeated edges across repetitions and is not an independent sample size. DA is averaged over nonempty repetitions. POI seen/unseen refers to the corresponding training pool; |y|≥1 membership uses 10^−12 tolerance, and strata can overlap. Raw/calibrated RMSE and repetition values are in local companions. Empty strata have NA metrics; no signal-to-noise ratio is inferred.", parts)

    estimates = sm["primary_estimates.csv"]
    require(len(estimates) == len(STATISTICS) and {r["statistic"] for r in estimates} == set(STATISTICS), "contrast population differs")
    rows, support = [], []
    for statistic, label in STATISTICS.items():
        r = one(estimates, statistic=statistic)
        require(int(r["valid_draws"]) + int(r["invalid_draws"]) == int(r["draws_requested"]), "bootstrap valid/invalid counts do not close")
        require(int(r["component_count"]) == summary["actual_prediction_components"], "bootstrap component scope differs")
        rows.append([label, fmt(r["estimate"]), fmt(r["lower_95"]), fmt(r["upper_95"])])
        support.append([label, n(r["draws_requested"]), n(r["valid_draws"]), n(r["invalid_draws"])])
    table(8, "Representation and strategy contrasts with conditional intervals.", f"All ratios use natural logarithms: γ=log(MAE_count/MAE_binary), θ=log(MAE_Direct/MAE_E1), θ_match=log(MAE_E0-matched/MAE_E1), θ_support=log(MAE_E0/MAE_E0-matched), and δ=θ_count−θ_binary. Negative γ favors count; positive θ or θ_match favors E1. Folds pool within each repetition before equal learner/repetition weighting; C1 uses RF/XGB only, and its two rows are exploratory. Intervals are conditional fixed-prediction source-plus-shared-identity component percentiles from {summary['actual_prediction_components']} contributing components, seed {cfg['estimands']['conditional_bootstrap']['seed']}; actual valid and invalid draws appear in Panel B. Zero MAE makes its log contrast NA, without an epsilon or silently dropping cells. Absolute errors/differences remain in companion tables. No family adjustment, total-uncertainty interval or equivalence test is added.", ["**Panel A. Contrasts and intervals.**\n\n" + panel(8, "a", ["Statistic", "Estimate", "Lower 95%", "Upper 95%"], rows, [S("primary_estimates.csv")]), "**Panel B. Actual conditional bootstrap support.**\n\n" + panel(8, "b", ["Statistic", "Draws requested", "Valid draws", "Invalid draws"], support, [S("primary_estimates.csv")])])

    boundary = sorted_model(sm["selection_boundary_summary.csv"])
    table(9, "Candidate selections at grid endpoints.", "Counts refer to the 15 final models per representation–learner–arm. At min/max means the tested numerical endpoint; every selection on a two-level parameter is an endpoint. These counts do not establish optimization failure or prove that a wider grid improves prediction. Full candidates and scales remain in local companions.", [panel(9, "a", ["Repr.", "Learner", "Arm", "Parameter", "Models", "At min", "At max"], [model_labels(r) + [r["parameter"], n(r["outer_models"]), n(r["at_min"]), n(r["at_max"])] for r in boundary], [S("selection_boundary_summary.csv")])])

    parts = []
    for label, name, letter in [("Internal", "main_consistency_learner_mean.csv", "a"), ("Exploratory external ensemble", "external_consistency.csv", "b")]:
        rows = sorted_model(dm[name])
        require({model_key(r) for r in rows} == combinations and len(rows) == len(combinations), "projection model scope differs")
        parts.append(f"**{label}.**\n\n" + panel(10, letter, ["Repr.", "Learner", "Arm", "κ cal", "MAE cal", "MAE projected", "RMSE projected"], [model_labels(r) + [fmt(r[k]) for k in ("calibrated_kappa", "calibrated_mae", "calibrated_projected_mae", "calibrated_projected_rmse")] for r in rows], [D(name)]))
    table(10, "Graph consistency and error after label-free projection.", "κ is the L2 residual-to-prediction norm ratio on the same evaluation graph; a zero norm gives κ=0. Record-node incidence projection uses no held-out truth and stays within each repeat/fold/condition block. Internal metrics pool five folds per repetition and then average repetitions; external metrics describe the fixed 21-edge exploratory ensemble. Least-squares projection does not guarantee a MAE benefit. Conservative-arm numerical residues are retained rather than treated as substantive inconsistency.", parts)

    stability = sorted(dm["E1_final_iteration5_summary.csv"], key=lambda r: (r["representation"] == "count", list(LEARNERS).index(r["learner"])))
    require(len(stability) == 4 and all(int(r["final_models"]) == 15 for r in stability), "tree E1 diagnostic scope differs")
    rows = []
    for r in stability:
        cells = [f"{fmt(r[field+'_median'])} [{fmt(r[field+'_min'])}, {fmt(r[field+'_max'])}]" for field in ("training_h_change_rmse", "alpha_change_record_weighted_rmse")]
        rows.append([REPRESENTATIONS[r["representation"]], LEARNERS[r["learner"]], *cells])
    table(11, "Tree E1 changes between the fourth and fifth fixed training rounds.", "Each row summarizes 15 final outer models by median [minimum, maximum]. Prediction changes use record-wise RMSE; offset changes use record-weighted block-offset RMSE, in pDC50. These are training diagnostics without held-out labels. Nonzero fifth-round changes do not establish convergence or select a different iteration count.", [panel(11, "a", ["Repr.", "Learner", "Prediction-change RMSE median [min, max]", "Offset-change RMSE median [min, max]"], rows, [D("E1_final_iteration5_summary.csv")])])

    raw_classes = dm["raw_representation_classes.csv"]
    parts = [panel(12, "a", ["Population / representation", "Records", "IDs", "Input classes", "Mixed-ID classes", "Records in mixed classes"], [[r["population_representation"].replace("_", " ")] + [n(r[k]) for k in ("records", "chemical_identities", "representation_classes", "classes_with_distinct_chemical_identities", "records_in_cross_identity_classes")] for r in raw_classes], [D("raw_representation_classes.csv")])]
    eff = dm["effective_input_scope_summary.csv"]
    rows = []
    tests = sorted([r for r in eff if r["population"] == "main" and r["scope"] == "outer_test"], key=lambda r: (int(r["repeat"]), int(r["outer_fold"])))
    require(len(tests) == 15, "effective test input scope differs")
    for r in tests:
        ext = one(eff, population="external", scope="all", repeat=r["repeat"], outer_fold=r["outer_fold"])
        rows.append([f"R{r['repeat']}/F{r['outer_fold']}", n(r["vocabulary_size"]), n(r["records"]), fmt(r["unknown_count_fraction_mean"], 4), fmt(r["unknown_count_fraction_max"], 4), fmt(ext["unknown_count_fraction_mean"], 4), fmt(ext["unknown_count_fraction_max"], 4)])
    parts.append("**Effective count mapping.**\n\n" + panel(12, "b", ["Outer split", "Vocab.", "Test rec.", "Unknown count mean", "Test maximum", "External mean", "External maximum"], rows, [D("effective_input_scope_summary.csv")]))
    bounds = sorted(dm["outer_effective_forced_zero_bounds.csv"], key=lambda r: (r["representation"] == "count", int(r["repeat"]), int(r["outer_fold"])))
    parts.append("**Actual test zero-difference support.**\n\n" + panel(12, "c", ["Repr.", "Outer split", "Test edges", "Exact d=0", "Forced-zero MAE contribution"], [[REPRESENTATIONS[r["representation"]], f"R{r['repeat']}/F{r['outer_fold']}"] + [n(r["test_pairs"]), n(r["test_zero_difference_pairs"]), fmt(r["test_forced_zero_mae_contribution"])] for r in bounds], [D("outer_effective_forced_zero_bounds.csv")]))
    all_main = [r for r in eff if r["population"] == "main" and r["scope"] == "all"]
    all_ext = [r for r in eff if r["population"] == "external"]
    table(12, "Raw representation equality and actual training-vocabulary constraints.", f"Raw classes group identical record vectors irrespective of identity; mixed-ID classes contain more than one full chemical identity. Across the 15 actual outer count vocabularies, the {desc['records']}-record pool has {value_range(r['effective_representation_classes'] for r in all_main)} input classes and {value_range(r['classes_with_distinct_chemical_identities'] for r in all_main)} mixed-ID classes; the seven external records have {value_range(r['effective_representation_classes'] for r in all_ext)} classes. Effective-zero-vector counts remain in companions. Unknown count fractions are per-record fractions of original environment multiplicities discarded, reported as their mean and maximum. Fold-level forced-zero contributions apply only to each actual test scope; raw full-graph zero-difference counts are not substituted for these constraints.", parts)

    pooled = [r for r in dm["posthoc_internal_group_census.csv"] if int(r["repeat"]) == -1 and int(r["outer_fold"]) == -1]
    require(len(pooled) == 3 and {r["group"] for r in pooled} == set(GROUPS), "internal group census differs")
    table(13, "Internal post hoc input-group support.", "Membership uses exact d equality in actual binary and count inputs, rather than fitted zeros. Persistent zero: both d=0; restored: binary d=0/count d≠0; both nonzero: both d≠0. Counts combine repeated test sets. IDs, records, blocks and dependency components may overlap across groups and are not independent sample sizes.", [panel(13, "a", ["Group", "Obs.", "Different edges", "IDs", "Records", "Blocks", "Dependency components"], [[r["group"].replace("_", " ")] + [n(r[k]) for k in ("observations", "unique_edges", "chemical_identities", "records", "condition_blocks", "accepted_internal_dependency_components")] for r in pooled], [D("posthoc_internal_group_census.csv")])])

    def contributions(sid, name, title, caption):
        rows = sorted_model(dm[name])
        expected = {(l, a) for _, l, a in combinations}
        require(len(rows) == len(expected) and {(r["learner"], r["pipeline"]) for r in rows} == expected, "posthoc arm scope differs")
        table(sid, title, caption, [panel(sid, "a", ["Learner", "Arm", "Persistent zero", "Restored", "Both nonzero", "Overall ΔMAE"], [[LEARNERS[r["learner"]], ARMS[r["pipeline"]]] + [fmt(r[k]) for k in GROUPS + ["overall_count_minus_binary_mae"]] for r in rows], [D(name)])])

    contributions(14, "posthoc_internal_calibrated_loss_contributions.csv", "Internal post hoc contributions to the representation MAE difference.", "Entries are the count−binary absolute-loss-sum difference divided by each repetition's total test-edge observations N_r, then averaged equally across repetitions. Negative values favor count. Group contributions sum to the overall ΔMAE; unweighted group MAEs do not. Empty groups have NA own-group metrics and zero contribution when the full scope is nonempty. Persistent-zero residues remain at companion precision. This input-defined decomposition does not identify a causal representation effect.")

    source_rows = []
    target_labels = {0: "IRAK4", 1: "TBK1 (training-unseen)", 3: "CDK6"}
    for r in source:
        c = one(census, scope="source", inner_fold=-1, outer_fold=r["source_index"])
        source_rows.append([f"S{r['source_index']}", r["source_id"].replace("PATENT:", "").replace("DOI:", "").replace("PMID:", "PMID "), target_labels[r["source_index"]], n(r["records"]), n(r["identities"]), n(c["training_records"]), n(c["training_identities"]), n(c["train_pairs"]), n(r["test_edges"]), n(r["strict_identity_unseen_edges"])])
    table(15, "Three fixed binary source holdouts and their support.", "S0, S1 and S3 retain original source indices 0/1/3 and seeds; S2 is retired without replacement or re-ranking. Each held-out cluster is excluded from fitting, inner selection, calibration and E1 offsets. The other two retained selected sources remain in training for that individual holdout. Both-identity-unseen means neither endpoint identity appears in any training record, and uses the same fitted model. Sources were fixed by graph/target/shared-identity coverage rather than sampled from a source population. Only binary inputs and original arms are fitted; S3 includes one-identity-seen edges and S1 has a training-unseen target.", [panel(15, "a", ["Source", "Identifier", "POI", "Source rec.", "Source IDs", "Train rec.", "Train IDs", "Train edges", "Test edges", "Both IDs unseen"], source_rows, [local(bundle, spec["source_holdout"]["selected_sources"]), census_path])])

    source_metrics = sm["source_metrics.csv"]
    require(not any(r["pipeline"] == "endpoint_E0_matched" for r in source_metrics), "matched source results unplanned")
    def source_error_rows(holdout, scenario, label):
        rows = []
        for learner in LEARNERS:
            key = {"holdout_id": holdout, "scenario": scenario, "learner": learner}
            chosen = {arm: one(source_metrics, **key, pipeline=arm, scale="calibrated") for arm in ("direct_pair", "endpoint_E0", "endpoint_E1")}
            if learner != "ridge":
                chosen["context_pair_C1"] = one(source_metrics, **key, pipeline="context_pair_C1", scale="calibrated")
            zero = one(source_metrics, **key, pipeline="zero", scale="raw")
            rows.append([label, LEARNERS[learner], n(zero["n"])] + [fmt(chosen[arm]["mae"]) if arm in chosen else "NA" for arm in ("direct_pair", "endpoint_E0", "endpoint_E1", "context_pair_C1")] + [fmt(zero["mae"])])
        return rows
    rows = [row for r in source for row in source_error_rows(r["holdout_id"], "source_unseen", f"S{r['source_index']}")]
    require(len(rows) == 9, "source learner rows differ")
    table(16, "All nine learner–source primary comparisons against Zero.", "Calibrated MAE uses all source-unseen edges, in pDC50. Zero predicts zero without calibration. C1 is defined only for RF/XGB, so Ridge C1 is NA. Each source has one selected fit per original arm and uses other-source inner predictions for tuning/calibration. There are no count or E0-matched source fits. These descriptive source cells are not independent trials.", [panel(16, "a", ["Source", "Learner", "Edges", "Direct", "E0", "E1", "C1", "Zero"], rows, [S("source_metrics.csv")])])

    s3 = next(r for r in source if r["source_index"] == 3)
    rows = source_error_rows(s3["holdout_id"], "source_identity_unseen", "Both unseen") + source_error_rows(s3["holdout_id"], "one_identity_seen", "One seen")
    table(17, "Identity visibility subsets of the same S3 source-held-out models.", f"S3 is the CDK6 DOI 10.1021/acs.jmedchem.9b00871 source. Its {s3['test_edges']} edges partition into {s3['strict_identity_unseen_edges']} both-unseen and {s3['one_identity_seen_edges']} one-seen edges. Values are calibrated MAE with subset-specific Zero, in pDC50. No extra identity-removal refit is made. Shared endpoints and small one-seen support limit causal interpretation.", [panel(17, "a", ["Subset", "Learner", "Edges", "Direct", "E0", "E1", "C1", "Zero"], rows, [S("source_metrics.csv")])])

    recurrence_rows = []
    for filename, label in [("revised_entity_source_comparisons.csv", "Same-identity/POI records"), ("repeated_pair_source_comparisons.csv", "Repeated entity/POI pairs")]:
        proof = desc["inherited_evidence_filter"][filename]
        rec = desc["recurrence"][filename]
        actual = builder.read(ddp / filename)
        require(proof["retained_comparisons"] == rec["n"] == len(actual), "current recurrence support differs")
        recurrence_rows.append([label, str(rec["n"]), "; ".join(f"{key}: {count}" for key, count in rec["condition_relation"].items()), "; ".join(f"{key}: {count}" for key, count in rec["normalized_source_relation"].items()), "0"])
    agg = next(row for row in historical if row[0] == "Retained endpoint / aggregation layer")
    recurrence_rows.extend([["Retained endpoint aggregation", agg[1], "One input row per retained key", "Upstream condition aggregation can remain", "Not established"], ["Independent run / replicate identifiers", "Not present in raw schema", "No run-level matching", "Not recoverable from reference equality", "Not established"]])
    table(18, "Repeated-record comparability and unresolved measurement-noise support.", f"The current {desc['records']}-record pool and {desc['pairs']}-edge graph retain all {desc['recurrence']['revised_entity_source_comparisons.csv']['n']} recurring identity–POI record comparisons and {desc['recurrence']['repeated_pair_source_comparisons.csv']['n']} entity–POI pair comparisons. The repaired source-row mapping preserves the 121 record comparisons and adds two pair comparisons, giving 558; the current condition/source relations are recomputed on the corrected graph and documented in DATA_DESCRIPTION.json. Normalized attribution merges verified aliases. Complete same-condition comparable remeasurements remain absent, so independent assay noise, repeatability and a general prediction limit cannot be estimated.", [panel(18, "a", ["Comparison", "Total", "Conditions", "Normalized source relation", "Complete same-condition repeats"], recurrence_rows, [ddp / "DATA_DESCRIPTION.json", ddp / "revised_entity_source_comparisons.csv", ddp / "repeated_pair_source_comparisons.csv", prior_path])])

    rows = [[r["compound_id"], r["PEG_n"], n(r["DC50_nM"]), fmt(r["pdc50_median"]), n(r["Dmax_percent"]), f"{n(r['baseline_exact_full_graph_match_rows'])} exact full-graph matches"] for r in external]
    table(19, "The fixed seven-entity BRD4 literature series.", "Compounds 6–12 share BRD4/CRBN, HEK293 endogenous HiBiT and 5 h conditions in DOI 10.1038/s41467-026-70663-1. Structures are manually transcribed molecular graphs, not author-released SMILES. Reported DC50 is in nM; z=9−log10(DC50 in nM). Seven entities define 21 shared-endpoint pairs. Controls 13/14 have no continuous values assigned. The source reports n=3 generally, n=2 for compound 8's HiBiT entry, and SEM within 0.2 log units without individual SEM values. This previously examined case remains exploratory.", [panel(19, "a", ["Compound", "PEG n", "DC50 (nM)", "pDC50", "Dmax (%)", "Chemical identity overlap with frozen TACK"], rows, [local(bundle, spec["external"]["records"])])])

    em = sm["external_metrics.csv"]
    calibrated = sorted_model([r for r in em if r["scale"] == "calibrated" and r["pipeline"] != "zero"])
    require(len(calibrated) == len(combinations) and {model_key(r) for r in calibrated} == combinations, "external ensemble scope differs")
    rows = []
    for r in calibrated:
        raw = one(em, **dict(zip(("representation", "learner", "pipeline"), model_key(r))), scale="raw")
        require(int(r["n"]) == 21, "external metric denominator differs")
        rows.append(model_labels(r) + [fmt(raw["mae"]), fmt(r["mae"]), fmt(r["rmse"]), fmt(r["DA_strict"], 4), fmt(r["DA_half"], 4), n(r["zero_predictions"])])
    zero = one(em, representation="binary", learner="ridge", pipeline="zero", scale="raw")
    rows.append(["Common", "—", "Zero", fmt(zero["mae"]), fmt(zero["mae"]), fmt(zero["rmse"]), fmt(zero["DA_strict"], 4), fmt(zero["DA_half"], 4), n(zero["zero_predictions"])])
    table(20, "Every exploratory external ensemble result with the common Zero baseline.", f"All {len(combinations)} representation–learner–arm ensembles use the same 21 pairs and 15 current outer final models. Each member applies its own training vocabulary and inner scale before predictions are averaged. Errors use pDC50. Strict DA gives a numerical-zero prediction zero credit; half DA gives 0.5, with |q|≤10^−12. Model performance does not upgrade this previously examined seven-entity series to a new independent validation source.", [panel(20, "a", ["Repr.", "Learner", "Arm", "MAE raw", "MAE cal", "RMSE cal", "DA strict", "DA half", "Zero preds"], rows, [S("external_metrics.csv")])])

    pc = [r for r in dm["pair_input_class_census.csv"] if r["population"] == "external"]
    inputs = []
    for rep in REPRESENTATIONS:
        pair_rows = [r for r in pc if r["representation"] == rep]
        direct = [r for r in pair_rows if r["pipeline"] == "direct_pair"]
        c1 = [r for r in pair_rows if r["pipeline"] == "context_pair_C1"]
        require(len(direct) == len(c1) == 15 and all(int(r["pairs"]) == 21 for r in pair_rows), "pair input member scope differs")
        if rep == "binary":
            record_classes = n(one(raw_classes, population_representation="external_binary")["representation_classes"])
        else:
            record_classes = value_range(r["effective_representation_classes"] for r in all_ext)
        inputs.append([REPRESENTATIONS[rep], record_classes, value_range(r["canonical_input_classes"] for r in direct), value_range(r["canonical_input_classes"] for r in c1), value_range(r["exact_zero_difference_pairs"] for r in direct)])
    ed, md = sorted_model(dm["external_ensemble_prediction_diagnostics.csv"]), dm["external_member_prediction_diagnostics.csv"]
    require(len(ed) == len(combinations) and {model_key(r) for r in ed} == combinations, "output diagnostic ensemble scope differs")
    rows = []
    for r in ed:
        members = [x for x in md if model_key(x) == model_key(r)]
        require(len(members) == 15 and {(int(x["repeat"]), int(x["outer_fold"])) for x in members} == {(i, j) for i in range(3) for j in range(5)}, "output diagnostic member scope differs")
        zeros = [int(x["calibrated_zero_predictions"]) for x in members]
        rows.append(model_labels(r) + [n(r["calibrated_zero_predictions"]), n(r["calibrated_distinct_prediction_classes_tol"]), str(sum(x > 0 for x in zeros)), str(sum(x == 21 for x in zeros))])
    table(21, "Actual external input classes and member/ensemble output zeros.", f"Panel A gives actual record X and canonical-edge Direct d/C1 [d,s] classes across the 15 current member vocabularies; a range indicates member variation. Endpoint E0/E0-matched/E1 inputs are record X. Panel B covers all {len(combinations)} ensembles and their members. Numerical zeros use 10^−12; output value classes have maximum within-class span ≤10^−12. Exact input equality and fitted output equality are distinct; all-zero members are retained.", [panel(21, "a", ["Repr.", "Record X classes", "Direct d classes", "C1 [d,s] classes", "Exact d=0 edges"], inputs, [D("pair_input_class_census.csv"), D("raw_representation_classes.csv"), D("effective_input_scope_summary.csv")]), panel(21, "b", ["Repr.", "Learner", "Arm", "Ensemble zeros", "Output classes", "Members with zeros /15", "All-zero members /15"], rows, [D("external_ensemble_prediction_diagnostics.csv"), D("external_member_prediction_diagnostics.csv")])])

    contributions(22, "posthoc_external_calibrated_loss_contributions.csv", "External ensemble post hoc contributions to the representation MAE difference.", "Entries are the count−binary absolute-loss-sum difference divided by the fixed 21 external edges after averaging member predictions. Negative values favor count. Group contributions sum to overall ΔMAE; unweighted group MAEs do not. This input-defined decomposition does not identify a causal representation effect.")
    ec = dm["posthoc_external_ensemble_group_census.csv"]
    require(len(ec) == 3 and sum(int(r["observations"]) for r in ec) == 21 and {r["group"] for r in ec} == set(GROUPS), "external group census differs")
    extra = panel(22, "b", ["Group", "Edges", "IDs", "Records", "Blocks", "External series components"], [[r["group"].replace("_", " ")] + [n(r[k]) for k in ("observations", "chemical_identities", "records", "condition_blocks", "external_series_components")] for r in ec], [D("posthoc_external_ensemble_group_census.csv")])
    sid, section = builder.sections[-1]
    builder.sections[-1] = (sid, section + "\n\n" + extra + "\n\nEmpty groups have NA own-group metrics; their zero loss and contribution mean no observations. The external series is one previously examined source with shared endpoints. Internal dependency IDs do not apply. Member loss is kept distinct from the loss of averaged ensemble predictions.\n")
    return builder.finish(spec, summary, diag)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--data-description", type=Path, required=True)
    parser.add_argument("--prior-si", type=Path, required=True, help="read-only R4 Supporting_Information.md; historical construction rows only")
    parser.add_argument("--output", type=Path, help="new directory inside this script's responsibility directory")
    parser.add_argument("--all-results-ready", action="store_true", help="explicit owner authorization after completed outputs")
    args = parser.parse_args()
    try:
        result = build(args)
    except (RuntimeError, FileNotFoundError, KeyError, ValueError) as error:
        print(json.dumps({"status": "SI_TABLE_BUILD_CLOSED", "reason": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
