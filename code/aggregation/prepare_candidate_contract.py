"""Prepare one isolated R5 candidate contract and RIPK1 data derivatives.

No training workers, estimators, model files, or production outputs are loaded.
The original entry is copied unchanged and remains closed by candidate status.
"""
from __future__ import annotations

import ast
import copy
import csv
import importlib.util
import itertools
import json
import math
import hashlib
import shutil
import sys
from collections import Counter
from pathlib import Path
import numpy as np

PROJECT = Path("/LOCAL_PROJECT_ROOT")
PREP = PROJECT / "intake/remaining_local_preparation_20261012/candidate_repair/r5_dependency_preparation"
OLD = PROJECT / "r5_analysis/repro/portable_training"
INPUTS = PREP / "isolated_r5_inputs"
SHADOW = PREP.parent / "impact/run_source_graphs_20261012/repair_seven_source_graphs"
CASE = PROJECT / "r5_analysis/admission_followup/schema_compatibility"
TARGET = PREP / "candidate_r5_contract"
STATUS = "PREPARED_CANDIDATE_NOT_EXECUTION_FROZEN"


def require(ok, reason):
    if not bool(ok):
        raise RuntimeError(reason)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def table(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def digest_new(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def destination(relative):
    path = TARGET / relative
    require(path.resolve().is_relative_to(TARGET.resolve()), "Write escapes candidate root")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def dump(relative, value):
    path = destination(relative)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return digest_new(path)


def write_table(relative, rows):
    require(rows, "Empty derivative CSV")
    path = destination(relative)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return digest_new(path)


def save_npz(relative, **arrays):
    path = destination(relative)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    return digest_new(path)


def truth(value):
    return str(value).lower() in ("true", "1")


def pure_function(path, name, namespace):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name], {"source": str(path), "definition": name, "lines": [node.lineno, node.end_lineno],
                             "execution": "unchanged AST-selected pure definition only"}


def main():
    require(Path(__file__).resolve().parent == PREP.resolve(), "Wrong preparation root")
    require(not TARGET.exists() and not TARGET.is_symlink(), "Refusing existing candidate contract directory")
    prior = read_json(INPUTS / "INPUT_PREPARATION_RECEIPT.json")
    require(prior["all_physical_members_labels_sources_partitions_preserved"], "Missing fixed-member preparation")
    require(read_json(PREP / "ROOT_FIXED_SPLIT_INTERPRETATION.json")["status"] ==
            "ROOT_DECIDED_FIXED_PHYSICAL_MEMBERSHIP_IDENTITY_ALIAS_TRANSPORT", "Missing root decision")
    historical_freeze = read_json(OLD / "freeze_receipt.json")
    independent = read_json(PROJECT / "remaining_preparation/independent_candidate_impact/FINAL_SHADOW_TABLES_CHECKS.json")
    require(independent["status"] == "PASS_INDEPENDENT_CURRENT_LAYER_SHADOW_CHECK", "Unaccepted shadow")
    shadow_sha = {Path(row["path"].replace("\\", "/")).resolve(): row["sha256"] for row in independent["input_hashes"]}
    old_spec = read_json(OLD / "spec.json")
    old_cfg = read_json(OLD / old_spec["scientific_config"])
    require(len(old_spec["jobs"]) == 420 and old_spec["expected_successful_fits"] == 17925, "Original scope drift")
    TARGET.mkdir(exist_ok=False)
    copied = {}

    def copy_bound(source, relative, expected=None, authority=None):
        source = Path(source).resolve(strict=True)
        require(source.is_file() and source.is_relative_to(PROJECT.resolve()), "Source escapes project")
        target = destination(relative)
        require(not target.exists(), "Refusing copied-file overwrite")
        shutil.copyfile(source, target)
        actual = digest_new(target)
        require(expected is None or actual == expected.upper(), "New copy differs from inherited binding: " + relative)
        copied[relative] = {"sha256": actual, "source_path": source.as_posix(),
                            "sha_authority": authority or "capture_new_copy", "bytes": target.stat().st_size}
        return actual

    for name in ["records_revised.csv", "pairs_revised.csv", "fingerprints.csv", "main_raw_sparse_counts.json"]:
        copy_bound(SHADOW / name, "input_build/data/" + name, shadow_sha[(SHADOW / name).resolve()],
                   "accepted FINAL_SHADOW_TABLES_CHECKS.json:input_hashes")
    for relative in ["data/outer_identity_folds.csv", "config/inner_identity_folds.csv",
                     "config/source_inner_identity_folds.csv", "data/record_sources.csv", "config/vocab_manifest.json"]:
        copy_bound(INPUTS / relative, "input_build/" + relative, prior["new_small_artifacts"][relative],
                   "isolated_r5_inputs/INPUT_PREPARATION_RECEIPT.json:new_small_artifacts")
    vocab_manifest = read_json(INPUTS / "config/vocab_manifest.json")
    require(len(vocab_manifest) == 60, "Missing R5 vocabulary layers")
    for layer, binding in vocab_manifest.items():
        copy_bound(INPUTS / binding["path"], "input_build/data/vocab/" + layer + ".npz", binding["sha256"],
                   "isolated_r5_inputs/config/vocab_manifest.json")
    original_relatives = ["entry.py", "contract.py", "run_queue.py", "SCIENTIFIC_INVARIANTS.json",
                          "engine/run_binary.py", "engine/run_count.py", "engine/model_adapters.py",
                          "engine/count_features.py", "engine/run_source.py",
                          "input_build/config/selected_sources.json", "protocol/EXECUTION_PROTOCOL.md",
                          "old_exploratory_case/CASE_SCOPE.json", "old_exploratory_case/records.csv",
                          "old_exploratory_case/pairs.csv", "old_exploratory_case/fingerprints.csv",
                          "old_exploratory_case/raw_sparse_counts.json"]
    for relative in original_relatives:
        copy_bound(OLD / relative, relative, historical_freeze["files"][relative], "original R5 freeze_receipt.json:files")
    cfg = copy.deepcopy(old_cfg)
    cfg["status"] = STATUS
    cfg["data"]["compatible_pairs"] = 5804
    require(all(cfg[key] == value for key, value in read_json(OLD / "SCIENTIFIC_INVARIANTS.json").items()),
            "Original scientific invariants changed")
    cfg_sha = dump("scientific_config.json", cfg)
    spec = copy.deepcopy(old_spec)
    spec.update(status=STATUS, run_id="R5_SOURCE_GRAPH_REPAIR_FIXED_MEMBERS_20261012_CANDIDATE",
                freeze_receipt="EXECUTION_FREEZE_PENDING.json", review_decision="protocol/CURRENT_PREPARATION_STATUS.md")
    spec_sha = dump("spec.json", spec)
    aliases = read_json(INPUTS / "IDENTITY_RECORD_ALIASES.json")
    alias_by_old = {row["original_record_id"]: row["record_id"] for row in table(SHADOW / "RECORD_COMPONENT_MAPPING.csv")}
    old_records = sorted(table(OLD / old_spec["inputs"]["records"]), key=lambda x: x["record_id"])
    records = sorted(table(TARGET / spec["inputs"]["records"]), key=lambda x: x["record_id"])
    old_by_id = {row["record_id"]: row for row in old_records}
    by_id = {row["record_id"]: row for row in records}
    old_pairs = sorted(table(OLD / old_spec["inputs"]["pairs"]), key=lambda x: x["pair_id"])
    pairs = sorted(table(TARGET / spec["inputs"]["pairs"]), key=lambda x: x["pair_id"])
    old_fp = {row["record_id"]: int(row["fp_hex"], 16) for row in table(OLD / old_spec["inputs"]["fingerprints"])}
    fp = {row["record_id"]: int(row["fp_hex"], 16) for row in table(TARGET / spec["inputs"]["fingerprints"])}
    old_raw = {row["record_id"]: dict(row["counts"]) for row in read_json(OLD / old_spec["inputs"]["count_cache"])["records"]}
    raw = {row["record_id"]: dict(row["counts"]) for row in read_json(TARGET / spec["inputs"]["count_cache"])["records"]}
    require(len(records) == 699 and len(pairs) == 5804, "Candidate census drift")
    require(set(fp) == set(raw) == set(by_id), "Candidate feature row coverage")
    for row in pairs:
        a, b = by_id[row["record_id_i"]], by_id[row["record_id_j"]]
        y = float(b["pdc50_median"]) - float(a["pdc50_median"])
        require(abs(y + float(row["true_delta"])) <= 1e-12, "Pair canonical direction mismatch")
        require(truth(row["is_cliff"]) == (abs(y) >= 1 - 1e-12), "Pair cliff scope mismatch")
    revisions = table(OLD / old_spec["revision_table"])
    for row in aliases["corrected_record_aliases"]:
        revisions.append({"record_id": row["record_id"], "action": "candidate_source_graph_repair",
                          "field": "identity_hash_and_structure_derived_fields",
                          "old_value": old_by_id[row["original_record_id"]]["identity_hash"], "new_value": row["identity_hash"],
                          "evidence": "accepted CANDIDATE_PATCHES.json + CANDIDATE_ACCEPTANCE.json + FINAL_SHADOW_TABLES_CHECKS.json",
                          "decision": "LOCAL_CANDIDATE_ACCEPTED_NOT_PRODUCTION_APPLIED"})
    revision_sha = write_table("input_build/revision_table.csv", revisions)
    prep_receipt = {"status": "CANDIDATE_INPUTS_PREPARED_NOT_EXECUTION_FROZEN", "revision_table_sha256": revision_sha,
                   "outputs": {key: copied[spec["inputs"][key]] for key in ["records", "pairs", "fingerprints", "outer_folds", "inner_folds"]},
                   "fixed_member_interpretation": str(PREP / "ROOT_FIXED_SPLIT_INTERPRETATION.json"),
                   "original_split_algorithm_executions": 0, "corrected_hash_rerun_equivalence_established": False,
                   "observed_rerun_membership_difference": "unknown", "model_reuse_authorized": False,
                   "scientific_execution_started": False, "old_RUN_BINDING_or_freeze_copied": False}
    prep_sha = dump("input_build/input_preparation_receipt.json", prep_receipt)
    status_text = "# Current candidate input preparation\n\nRoot has decided to transport the original physical split memberships through the accepted one-to-one identity aliases. This is an isolated derivative of the original frozen R5 protocol, not a corrected-hash resplit. Original scientific invariants, pair-micro helper, seeds, grids, strategies and fixed case are retained.\n\nPreparation only. No execution freeze or RUN_BINDING exists, no queue is started, no models are loaded, and no new predictions are produced. The original input preparation and historical author/release/source gaps are not converted into new approvals. Execution binding and cache import are pending engineering integration; they do not reopen the decided split route.\n"
    destination(spec["review_decision"]).write_text(status_text, encoding="utf-8", newline="\n")
    # Use the copied standard-library contract only; scientific workers are not imported.
    module_spec = importlib.util.spec_from_file_location("candidate_r5_pure_contract", TARGET / "contract.py")
    contract = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(contract)
    schema_report = contract.inspect_inputs(*(TARGET / spec["inputs"][key] for key in
                                             ["records", "pairs", "fingerprints", "outer_folds", "inner_folds"]))
    try:
        contract.load_for_execution(TARGET / "spec.json")
    except contract.GateError as exc:
        closed_gate = str(exc)
    else:
        raise RuntimeError("Candidate execution gate unexpectedly open")
    require("not frozen/authorized" in closed_gate, "Wrong candidate closed-gate reason")
    for relative in ["entry.py", "contract.py", "run_queue.py", "engine/run_binary.py", "engine/run_count.py",
                     "engine/model_adapters.py", "engine/count_features.py", "engine/run_source.py"]:
        ast.parse((TARGET / relative).read_text(encoding="utf-8"))
    grid, grid_evidence = pure_function(OLD / "engine/run_binary.py", "candidate_grid", {"itertools": itertools})
    scale, scale_evidence = pure_function(OLD / "engine/model_adapters.py", "nonnegative_l1_scale", {"np": np, "math": math})
    grids = {learner: grid(cfg, learner) for learner in ["ridge", "random_forest", "xgboost"]}
    old_outer = table(OLD / old_spec["inputs"]["outer_folds"])
    outer = table(TARGET / spec["inputs"]["outer_folds"])
    old_inner = table(OLD / old_spec["inputs"]["inner_folds"])
    inner = table(TARGET / spec["inputs"]["inner_folds"])
    old_sources = {row["record_id"]: row["c2_source_id"] for row in table(OLD / old_spec["source_holdout"]["record_sources"])}
    sources = {row["record_id"]: row["c2_source_id"] for row in table(TARGET / spec["source_holdout"]["record_sources"])}
    old_source_inner = table(OLD / old_spec["source_holdout"]["inner_folds"])
    source_inner = table(TARGET / spec["source_holdout"]["inner_folds"])
    selection = read_json(TARGET / spec["source_holdout"]["selected_sources"])
    old_y = np.asarray([float(old_by_id[p["record_id_j"]]["pdc50_median"]) - float(old_by_id[p["record_id_i"]]["pdc50_median"]) for p in old_pairs])
    new_y = np.asarray([float(by_id[p["record_id_j"]]["pdc50_median"]) - float(by_id[p["record_id_i"]]["pdc50_median"]) for p in pairs])

    def e1_positions(rows, positions):
        keys = [rows[i]["normalized_poi_id"] + "|" + rows[i]["pair_context_hash"] for i in positions]
        counts = Counter(keys)
        kept = [i for i, key in zip(positions, keys) if counts[key] >= 2]
        return kept, {"retained_blocks": sum(n >= 2 for n in counts.values()),
                      "singleton_blocks_excluded": sum(n == 1 for n in counts.values())}

    def mask_scope(rows, edge_rows, rtrain, ptrain, predict):
        rpos = [i for i, r in enumerate(rows) if rtrain(r)]
        ppos = [i for i, p in enumerate(edge_rows) if ptrain(p)]
        vpos = [i for i, p in enumerate(edge_rows) if predict(p)]
        epos, estats = e1_positions(rows, rpos)
        require(rpos and ppos and vpos and epos, "Empty layer training/validation support")
        return {"r": rpos, "p": ppos, "v": vpos, "e": epos, "stats": estats}

    layers = {}
    for repeat in range(3):
        om_old = {r["identity_hash"]: int(r["outer_fold"]) for r in old_outer if int(r["repeat"]) == repeat}
        om = {r["identity_hash"]: int(r["outer_fold"]) for r in outer if int(r["repeat"]) == repeat}
        for fold in range(5):
            im_old = {r["identity_hash"]: int(r["inner_fold"]) for r in old_inner if int(r["repeat"]) == repeat and int(r["outer_fold"]) == fold}
            im = {r["identity_hash"]: int(r["inner_fold"]) for r in inner if int(r["repeat"]) == repeat and int(r["outer_fold"]) == fold}
            for phase in [-1, 0, 1, 2]:
                scopes = []
                for rows, edges, oo, ii in [(old_records, old_pairs, om_old, im_old), (records, pairs, om, im)]:
                    if phase == -1:
                        scope = mask_scope(rows, edges,
                            lambda r: oo[r["identity_hash"]] != fold,
                            lambda p: oo[p["identity_hash_i"]] != fold and oo[p["identity_hash_j"]] != fold,
                            lambda p: oo[p["identity_hash_i"]] == fold and oo[p["identity_hash_j"]] == fold)
                    else:
                        scope = mask_scope(rows, edges,
                            lambda r: r["identity_hash"] in ii and ii[r["identity_hash"]] != phase,
                            lambda p: p["identity_hash_i"] in ii and p["identity_hash_j"] in ii and ii[p["identity_hash_i"]] != phase and ii[p["identity_hash_j"]] != phase,
                            lambda p: p["identity_hash_i"] in ii and p["identity_hash_j"] in ii and ii[p["identity_hash_i"]] == phase and ii[p["identity_hash_j"]] == phase)
                    scopes.append(scope)
                layers[f"R{repeat}_O{fold}_I{phase}"] = {"kind": "internal", "repeat": repeat, "outer_fold": fold,
                                                         "phase": phase, "old": scopes[0], "new": scopes[1]}
    for selected in selection["source_holdouts"]:
        source_index, sid = int(selected["source_index"]), selected["source_id"]
        for phase in [-1, 0, 1, 2]:
            scopes = []
            for rows, edges, sm, sii in [(old_records, old_pairs, old_sources, old_source_inner), (records, pairs, sources, source_inner)]:
                ii = {r["identity_hash"]: int(r["inner_fold"]) for r in sii if int(r["source_index"]) == source_index}
                scope = mask_scope(rows, edges,
                    lambda r: sm[r["record_id"]] != sid and (phase == -1 or ii[r["identity_hash"]] != phase),
                    lambda p: sm[p["record_id_i"]] != sid and sm[p["record_id_j"]] != sid and (phase == -1 or ii[p["identity_hash_i"]] != phase and ii[p["identity_hash_j"]] != phase),
                    lambda p: (sm[p["record_id_i"]] == sid and sm[p["record_id_j"]] == sid) if phase == -1 else
                    (sm[p["record_id_i"]] != sid and sm[p["record_id_j"]] != sid and ii[p["identity_hash_i"]] == phase and ii[p["identity_hash_j"]] == phase))
                scopes.append(scope)
            layers[f"S{source_index}_I{phase}"] = {"kind": "source", "source_index": source_index, "phase": phase,
                                                  "old": scopes[0], "new": scopes[1]}

    def record_signature(rows, fmap, counts, positions, representation, environment_set, e1=False):
        result = []
        for i in positions:
            r = rows[i]
            features = fmap[r["record_id"]] if representation == "binary" else tuple(sorted((int(k), int(v)) for k, v in counts[r["record_id"]].items() if k in environment_set))
            row = (features, float(r["pdc50_median"]))
            if e1:
                row += (r["normalized_poi_id"], r["pair_context_hash"])
            result.append(row)
        return result

    def pair_signature(edge_rows, fmap, counts, ys, positions, representation, environment_set, c1=False):
        result = []
        for i in positions:
            p = edge_rows[i]
            ai, bj = p["record_id_i"], p["record_id_j"]
            if representation == "binary":
                a, b = fmap[ai], fmap[bj]
                d = (b & ~a, a & ~b)
                features = d + ((a & b, a ^ b) if c1 else ())
            else:
                a, b = counts[ai], counts[bj]
                keys = (set(a) | set(b)) & environment_set
                diff = tuple(sorted((int(k), int(b.get(k, 0) - a.get(k, 0))) for k in keys if b.get(k, 0) != a.get(k, 0)))
                features = (diff,)
                if c1:
                    features += (tuple(sorted((int(k), int(a.get(k, 0) + b.get(k, 0))) for k in keys)),)
            result.append((features, float(ys[i])))
        return result

    support_rows, mask_index, exact_inputs = [], {}, []
    for name, layer in layers.items():
        before, after = layer["old"], layer["new"]
        train_ids = [records[i]["record_id"] for i in after["r"]]
        require({old_records[i]["source_row_ids"] for i in before["r"]} == {records[i]["source_row_ids"] for i in after["r"]}, "Fixed physical training changed")
        if layer["kind"] == "internal":
            with np.load(TARGET / ("input_build/data/vocab/" + name + ".npz"), allow_pickle=False) as values:
                env = values["environment_ids"].tolist()
                require(values["training_record_ids"].tolist() == train_ids, "Count vocabulary differs from runner row order")
            with np.load(OLD / ("input_build/data/vocab/" + name + ".npz"), allow_pickle=False) as values:
                old_env = values["environment_ids"].tolist()
            reps = ["binary", "count"]
        else:
            env, old_env, reps = [], [], ["binary"]
        rel = "input_build/masks/" + name + ".npz"
        mask_sha = save_npz(rel, training_record_positions=np.asarray(after["r"], np.int64),
                           e1_training_record_positions=np.asarray(after["e"], np.int64),
                           training_pair_indices=np.asarray(after["p"], np.int64), predict_pair_indices=np.asarray(after["v"], np.int64),
                           training_record_ids=np.asarray(train_ids, dtype=str),
                           training_pair_ids=np.asarray([pairs[i]["pair_id"] for i in after["p"]], dtype=str),
                           predict_pair_ids=np.asarray([pairs[i]["pair_id"] for i in after["v"]], dtype=str))
        mask_index[name] = {"path": rel, "sha256": mask_sha, "row_order": "record_id/pair_id ascending, mergesort semantics"}
        train_g = {records[i]["normalized_poi_id"] for i in after["r"]}
        support_rows.append({"layer": name, "kind": layer["kind"], "phase": layer["phase"],
                             "old_training_records": len(before["r"]), "training_records": len(after["r"]),
                             "old_training_pairs": len(before["p"]), "training_pairs": len(after["p"]),
                             "old_predict_pairs": len(before["v"]), "predict_pairs": len(after["v"]),
                             "training_cliffs": sum(truth(pairs[i]["is_cliff"]) for i in after["p"]),
                             "predict_cliffs": sum(truth(pairs[i]["is_cliff"]) for i in after["v"]),
                             "e1_retained_records": len(after["e"]), **after["stats"],
                             "predict_pairs_seen_g": sum(pairs[i]["normalized_poi_id"] in train_g for i in after["v"]),
                             "predict_pairs_unseen_g": sum(pairs[i]["normalized_poi_id"] not in train_g for i in after["v"])})
        for rep in reps:
            columns_equal = rep == "binary" or env == old_env
            eset = set(env)
            oe = set(old_env)
            comparisons = {
                "endpoint_E0": record_signature(old_records, old_fp, old_raw, before["r"], rep, oe) == record_signature(records, fp, raw, after["r"], rep, eset),
                "endpoint_E0_matched": record_signature(old_records, old_fp, old_raw, before["e"], rep, oe) == record_signature(records, fp, raw, after["e"], rep, eset),
                "endpoint_E1": record_signature(old_records, old_fp, old_raw, before["e"], rep, oe, True) == record_signature(records, fp, raw, after["e"], rep, eset, True),
                "direct_pair": pair_signature(old_pairs, old_fp, old_raw, old_y, before["p"], rep, oe) == pair_signature(pairs, fp, raw, new_y, after["p"], rep, eset),
                "context_pair_C1": pair_signature(old_pairs, old_fp, old_raw, old_y, before["p"], rep, oe, True) == pair_signature(pairs, fp, raw, new_y, after["p"], rep, eset, True)}
            for pipeline, equal in comparisons.items():
                if layer["kind"] == "source" and pipeline == "endpoint_E0_matched":
                    continue
                exact_inputs.append({"layer": name, "representation": rep, "pipeline": pipeline,
                                     "column_order_equal": columns_equal, "ordered_numeric_training_input_equal": bool(columns_equal and equal),
                                     "method": "exact stored binary-bit/sparse-count row signatures and labels; no dense matrix or worker import"})
    dump("input_build/MASK_INDEX.json", mask_index)
    write_table("input_build/R5_LAYER_SUPPORT.csv", support_rows)
    dump("reuse_plan/EXACT_TRAINING_INPUT_COMPARISONS.json", exact_inputs)

    # S0 training and validation objects are unchanged; transport historical OOF indices explicitly.
    s0_layers = [name for name in layers if name.startswith("S0_")]
    require(all(row["ordered_numeric_training_input_equal"] for row in exact_inputs if row["layer"] in s0_layers), "S0 numeric training not exact")
    old_pair_lookup = {p["pair_id"]: i for i, p in enumerate(old_pairs)}
    new_pair_lookup = {p["pair_id"]: i for i, p in enumerate(pairs)}
    s0_oof_old = sorted(i for f in range(3) for i in layers[f"S0_I{f}"]["old"]["v"])
    s0_oof_new = sorted(i for f in range(3) for i in layers[f"S0_I{f}"]["new"]["v"])
    require([old_pairs[i]["pair_id"] for i in s0_oof_old] == [pairs[i]["pair_id"] for i in s0_oof_new], "S0 validation pair objects changed")
    require(np.array_equal(old_y[s0_oof_old], new_y[s0_oof_new]), "S0 validation targets changed")
    s0_members = []
    checked_candidates = 0
    for learner in ["ridge", "random_forest", "xgboost"]:
        pipelines = ["direct_pair", "endpoint_E0", "endpoint_E1"] + ([] if learner == "ridge" else ["context_pair_C1"])
        for pipeline in pipelines:
            historical = OLD / "run_output/source_holdout/results" / learner / "C2_S0" / pipeline
            selected = read_json(historical / "selection.json")
            require(len(selected["candidates"]) == 8, "S0 historical grid cardinality")
            candidate_checks = []
            for candidate in grids[learner]:
                cid = candidate["candidate_id"]
                row = next(x for x in selected["candidates"] if x["candidate_id"] == cid)
                require(row["params"] == candidate["params"], "Historical selected parameters differ from original grid")
                complete = read_json(historical / "candidates" / cid / "candidate_complete.json")
                require(complete == row, "S0 candidate summary differs from selection")
                oof_path = historical / "candidates" / cid / "oof_predictions.npz"
                with np.load(oof_path, allow_pickle=False) as values:
                    indices, predictions = values["pair_indices"], values["raw_prediction"]
                require(indices.tolist() == s0_oof_old and np.isfinite(predictions).all(), "Historical OOF scope/finite failure")
                new_indices = np.asarray([new_pair_lookup[old_pairs[i]["pair_id"]] for i in indices], np.int64)
                require(new_indices.tolist() == s0_oof_new, "Historical OOF transport order mismatch")
                computed_c = scale(old_y[indices], predictions)
                require(abs(computed_c - row["scale_c"]) <= 1e-12, "Historical R5 pair-micro calibration mismatch")
                calibrated_mae = float(np.mean(np.abs(old_y[indices] - row["scale_c"] * predictions)))
                require(abs(calibrated_mae - row["inner_oof_calibrated_mae"]) <= 1e-12, "Historical OOF objective mismatch")
                rel = f"reuse_plan/s0_historical_oof_reindexed/{learner}/{pipeline}/{cid}.npz"
                sha = save_npz(rel, old_pair_indices=indices, new_pair_indices=new_indices,
                               pair_ids=np.asarray([old_pairs[i]["pair_id"] for i in indices], dtype=str),
                               historical_raw_prediction=predictions)
                candidate_checks.append({"candidate_id": cid, "params": candidate["params"], "pair_rows": len(indices),
                                         "new_index_values_changed": int(np.sum(indices != new_indices)), "scale_c": computed_c,
                                         "calibration_and_objective_match": True, "path": rel, "sha256": sha,
                                         "source_path": oof_path.as_posix(), "new_model_prediction": False})
                checked_candidates += 1
            best = min(row["inner_oof_calibrated_mae"] for row in selected["candidates"])
            choice = min((row for row in selected["candidates"] if row["inner_oof_calibrated_mae"] <= best + 1e-12), key=lambda r: r["candidate_id"])
            require(choice == selected["selected"], "Historical S0 selected candidate mismatch")
            final = read_json(historical / "final/unit_complete.json")
            model = OLD / f"run_output/source_holdout/models/C2_{learner}/outer_0/{pipeline}.joblib"
            require(model.is_file() and Path(final["model_path"]).resolve() == model.resolve(), "Missing S0 permanent model")
            fit_receipts = final["fit_receipts"]
            expected_cost = 5 if pipeline == "endpoint_E1" and learner != "ridge" else 1
            require(final["fit_count"] == len(fit_receipts) == expected_cost, "S0 final fixed iteration mismatch")
            expected_rows = (2 * len(layers["S0_I-1"]["new"]["p"]) if pipeline in ["direct_pair", "context_pair_C1"] else
                             len(layers["S0_I-1"]["new"]["e"] if pipeline == "endpoint_E1" else layers["S0_I-1"]["new"]["r"]))
            require(all(receipt["rows"] == expected_rows for receipt in fit_receipts), "S0 recorded training row support mismatch")
            s0_members.append({"learner": learner, "pipeline": pipeline, "model_path": model.as_posix(),
                               "model_bytes": model.stat().st_size, "model_loaded": False, "model_sha256": None,
                               "artifact_parameter_introspection": "NOT_DONE_NO_MODEL_LOAD",
                               "artifact_identity_binding": "PENDING_EXACT_MODEL_BYTE_BINDING",
                               "training_numeric_input_order_and_support_exact": True, "seed": selection["model_seed"],
                               "selected_candidate_id": choice["candidate_id"], "selected_params": choice["params"],
                               "scale_c": choice["scale_c"], "historical_candidate_oof_checks": candidate_checks,
                               "reuse_eligible_dependency_class": "UNCHANGED_TRAINING_AND_OOF_REQUIRES_PERMANENT_MODEL_BINDING_AND_NEW_TEST_PREDICTION",
                               "new_test_pairs": len(layers["S0_I-1"]["new"]["v"]), "old_test_predictions_reusable": False})
    require(len(s0_members) == 11 and checked_candidates == 88, "Incomplete S0 proof")
    dump("reuse_plan/S0_EXACT_DEPENDENCY_AND_OOF_PROOF.json", {"status": "PASS_STORED_INPUT_GRID_OOF_AND_SELECTION_PROOF_MODEL_BINDING_PENDING",
         "members": s0_members, "historical_candidates_checked": 88, "historical_oof_pairs_each": len(s0_oof_old),
         "original_R5_helper": scale_evidence, "original_candidate_grid": grid_evidence,
         "model_reuse_authorized": False, "new_predictions": 0,
         "limits": ["No model bytes were loaded or hashed; serialized estimator parameters remain uninspected.",
                    "Historical raw OOF is reindexed only where original training and validation numeric rows are exact; not copied into run_output.",
                    "S0 corrected test graph has2222 edges; old final predictions cannot replace new model evaluation."]})

    # Fixed RIPK1 labels/structures/pairs remain original inputs; only applicability data are derived.
    case_records = sorted(table(CASE / "inputs/records.csv"), key=lambda r: r["record_id"])
    case_pairs = table(CASE / "inputs/pairs_all45.csv")
    case_primary = table(CASE / "inputs/pairs.csv")
    case_fps = {r["record_id"]: int(r["fp_hex"], 16) for r in table(CASE / "inputs/fingerprints.csv")}
    case_raw = {r["record_id"]: dict(r["counts"]) for r in read_json(CASE / "inputs/raw_sparse_counts.json")["records"]}
    require(len(case_records) == 10 and len(case_pairs) == 45 and len(case_primary) == 42, "Fixed RIPK1 case drift")
    require({p["pair_id"] for p in case_primary} == {p["pair_id"] for p in case_pairs if truth(p["primary_ge05"])}, "Fixed primary42 scope mismatch")
    require("NAME:RIPK1" not in {r["normalized_poi_id"] for r in records}, "RIPK1 POI not unseen")
    old_nn_path = PROJECT / "r5_analysis/ripk1_case/applicability_nearest699.csv"
    old_nn = {r["record_id"]: r for r in table(old_nn_path)} if old_nn_path.is_file() else {}
    neighbours = []
    for record in case_records:
        a = case_fps[record["record_id"]]
        scored = []
        for training in records:
            b = fp[training["record_id"]]
            union = (a | b).bit_count()
            require(union, "Zero nearest-neighbour union")
            scored.append(((a & b).bit_count() / union, training))
        maximum = max(value for value, _ in scored)
        nearest = next(r for value, r in scored if value == maximum)
        historical_nn = old_nn.get(record["record_id"], {})
        neighbours.append({"record_id": record["record_id"], "compound_id": record["compound_id"],
                           "nearest_training_record_id": nearest["record_id"], "nearest_training_identity_hash": nearest["identity_hash"],
                           "nearest_binary_tanimoto": maximum,
                           "nearest_similarity_bin": "<0.5" if maximum < .5 else "0.5-<0.7" if maximum < .7 else ">=0.7",
                           "identical_binary_feature_vector": any(a == value for value in fp.values()),
                           "training_population": "R5_candidate_fixed_physical699", "POI_status": "unseen",
                           "old_nearest_training_record_id": historical_nn.get("nearest_training_record_id", ""),
                           "old_nearest_binary_tanimoto": historical_nn.get("nearest_binary_tanimoto", "")})
    write_table("ripk1_data_preparation/applicability_nearest699.csv", neighbours)
    unknown_rows = []
    for layer in sorted(name for name in vocab_manifest if name.endswith("I-1")):
        with np.load(TARGET / ("input_build/data/vocab/" + layer + ".npz"), allow_pickle=False) as values:
            known = set(values["environment_ids"].tolist())
        for record in case_records:
            counts = case_raw[record["record_id"]]
            unknown = {k: v for k, v in counts.items() if k not in known}
            total = sum(counts.values())
            missing = sum(unknown.values())
            unknown_rows.append({"vocabulary_layer": layer, "record_id": record["record_id"],
                                 "unknown_environment_ids": len(unknown), "raw_environment_ids": len(counts),
                                 "unknown_environment_count": missing, "raw_environment_count": total,
                                 "unknown_id_fraction": len(unknown) / len(counts), "unknown_count_fraction": missing / total,
                                 "mapped_environment_count": total - missing})
    require(len(unknown_rows) == 150 and len(neighbours) == 10, "RIPK1 data derivative census")
    write_table("ripk1_data_preparation/count_unknown_environments.csv", unknown_rows)
    schema_receipt = read_json(CASE / "SCHEMA_RECEIPT.json")
    dump("ripk1_data_preparation/INPUT_AND_PARENT_BINDING_PLAN.json", {
         "status": "DATA_DERIVATIVES_PREPARED_420_PARENT_REBINDING_PENDING", "fixed_case_schema_directory": CASE.as_posix(),
         "fixed_case_existing_input_bindings": schema_receipt["input_bindings"], "fixed_case_inputs_modified": False,
         "fixed_records": 10, "primary_pairs": 42, "all_pairs": 45, "excluded_compounds": ["18c", "18d"],
         "nearest_training_records": 699, "unknown_coverage_rows": 150, "count_vocabularies": 15,
         "parent_members_pending": 420, "parent_combinations": 28, "member_specific_scale_before_mean": True,
         "original_33_source_completion_gate": "PENDING_MANDATORY_REBINDING_NOT_BYPASSED",
         "old_source_holdout_models_used_for_actual_case_prediction": False,
         "member_fields_pending": ["model_path", "model_sha256", "selected_candidate_id", "selected_params", "selection_path",
                                    "inner_oof_candidate_path", "inner_oof_predictions_path", "scale_c", "parent_run_id",
                                    "parent_spec_sha256", "parent_freeze_sha256"],
         "model_loads": 0, "fits": 0, "predictions": 0, "vocabulary_expansion": False,
         "nearest_rule": "sort corrected main record_id ascending; first exact maximum binary Tanimoto",
         "unknown_rule": "drop unknown environments; original fixed case raw counts are unchanged",
         "source_primary_truth_gaps": "Original AAM/VoR/N3 scope and source gaps unchanged"})
    dump("FIT_BUDGET_AND_REUSE_BOUNDARIES.json", {
         "status": "PREPARATION_ONLY_NO_QUEUE_NO_CACHE_IMPORT", "original_runner_fresh_internal_fits": 16500,
         "original_runner_fresh_source_fits": 1425, "original_runner_full_fresh_fits": 17925,
         "source_S1_S3_fresh_fits": 950, "S0_unchanged_dependency_class_fits": 475,
         "conditional_new_fit_budget_if_all11_S0_models_are_exactly_bound_and_OOF_import_is_implemented": 17450,
         "currently_execution_ready_reused_units": 0, "stock_entry_unmodified_requires_full_fresh_accounting": 17925,
         "internal_R2_O1_I1": "ARTIFACT_MISSING_REQUIRES_NEW_FIT_UNLESS_RETAINED_MODEL_PROVEN",
         "internal_unchanged_training_layer_does_not_prove_changed_validation_predictions_cached": True,
         "S0_candidate_OOF_units": 88 * 3, "S0_permanent_final_models": 11,
         "S0_test_prediction_requires_new_evaluation": True, "S0_model_byte_and_serialized_params_proof_pending": True,
         "no_old_c_or_selection_used_outside_proven_unchanged_S0_dependency_class": True,
         "source_completion_gate_pending_and_required": True,
         "count_formula": "internal45 inner layers*8 candidates*44 estimator fits +15 final layers*44 =16500; each source3 inner*8*19 +19 =475",
         "original_runner_reuse_integration_needed": "Original fit accounting expects all FIT_COMPLETE events; explicit reuse receipts/new binding must distinguish inherited events from fresh fits. No old ledger or RUN_BINDING is copied."})
    dump("INPUT_CONTRACT.json", {"schema": "R5_CANDIDATE_FIXED_MEMBER_INPUT_CONTRACT_V1", "status": STATUS,
         "candidate_spec_sha256": spec_sha, "candidate_scientific_config_sha256": cfg_sha, "input_preparation_receipt_sha256": prep_sha,
         "copied_files": copied, "sha_policy": "single readback of new copies against inherited SHA; no old source rehash",
         "parent_run_id": old_spec["run_id"], "parent_spec_sha256": historical_freeze["files"]["spec.json"],
         "root_fixed_member_decision": (PREP / "ROOT_FIXED_SPLIT_INTERPRETATION.json").as_posix(),
         "original_split_algorithm_executions": 0, "corrected_hash_rerun_equivalence_established": False,
         "observed_rerun_membership_difference": "unknown",
         "outer_lineage": "R5 legacy outer bytes match Route-BC1's1605 rows; these are a strict key/value subset of Route-S's1629 rows. Reuses accepted independent1468 fact checks.",
         "schema_report": schema_report, "execution_gate_closed_reason": closed_gate,
         "no_old_RUN_BINDING_or_freeze": not (TARGET / "RUN_BINDING.json").exists() and not (TARGET / "freeze_receipt.json").exists(),
         "model_loads": 0, "fits": 0, "new_predictions": 0,
         "scientific_release_ready": False, "production_applied": False,
         "next_engineering_step": "Concentrated acceptance, then explicit new execution binding and chosen full-fresh or proven-S0 cache integration; fixed split route is already decided."})
    require(not any(name in sys.modules for name in ["joblib", "sklearn", "xgboost", "rdkit", "run_binary", "run_count"]),
            "Scientific worker/model module unexpectedly imported")
    summary = {"status": STATUS, "candidate_root": TARGET.as_posix(), "copied_files": len(copied),
               "stored_pair_masks": len(mask_index), "all_mask_support_nonempty": True, "vocabularies": 60,
               "S0_historical_candidates_checked": checked_candidates, "S0_permanent_models_stat_only": len(s0_members),
               "RIPK1_nearest_rows": len(neighbours), "RIPK1_unknown_rows": len(unknown_rows),
               "model_loads": 0, "fits": 0, "new_predictions": 0, "execution_gate_closed": True}
    dump("PREPARATION_RECEIPT.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
