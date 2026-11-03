"""Read frozen R5 features and completed predictions; update existing diagnostics.

This command never imports a learner, loads a fitted object, fits, predicts,
selects, bootstraps, or changes a scientific input. Run only after the owner
authorizes the final consolidated analysis of completed training outputs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from diagnostic_math import (
    COMBO, GROUPS, KEY, POSTHOC, TOL, attach, census_tables, class_table,
    classes_summary, classify, difference_mean, direction_metrics, digest,
    grouped_metrics, map_counts, pair_input_census, pair_inputs, project_block,
    repeat_equal_summary, representation_differences, require, signature,
    zero_evidence,
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest().upper()


def local_path(root, relative):
    require(isinstance(relative, str) and bool(relative), "missing relative bundle path")
    path = Path(relative)
    require(not path.is_absolute() and not path.drive, "bundle paths must be relative")
    resolved = (root / path).resolve()
    require(resolved.is_relative_to(root), "bundle path escapes its root")
    return resolved


def as_bool(series):
    if series.dtype == bool:
        return series.to_numpy(bool)
    values = series.astype(str).str.lower()
    require(values.isin(["true", "false", "1", "0"]).all(), "invalid boolean field")
    return values.isin(["true", "1"]).to_numpy(bool)


@dataclass
class Population:
    records: pd.DataFrame
    pairs: pd.DataFrame
    X_record: np.ndarray
    raw_counts: list
    pair_i: np.ndarray
    pair_j: np.ndarray
    z: np.ndarray


def load_population(root, paths):
    records = pd.read_csv(local_path(root, paths["records"]), keep_default_na=False)
    if "primary_pair_eligible" in records:
        records = records.loc[as_bool(records.primary_pair_eligible)].copy()
    records = records.sort_values("record_id", kind="mergesort").reset_index(drop=True)
    pairs = pd.read_csv(local_path(root, paths["pairs"]), keep_default_na=False)
    pairs = pairs.sort_values("pair_id", kind="mergesort").reset_index(drop=True)
    fps = pd.read_csv(local_path(root, paths["fingerprints"]), keep_default_na=False)
    require(records.record_id.is_unique and pairs.pair_id.is_unique and fps.record_id.is_unique,
            "duplicate input record, pair, or fingerprint keys")
    fp_map = fps.set_index("record_id").fp_hex
    vectors = []
    for record_id in records.record_id:
        require(record_id in fp_map, "record fingerprint missing")
        value = fp_map.at[record_id]
        require(isinstance(value, str) and len(value) == 512 and value == value.upper(),
                "invalid fixed 2048-bit hexadecimal fingerprint")
        vectors.append(np.unpackbits(np.frombuffer(bytes.fromhex(value)[::-1], dtype=np.uint8),
                                     bitorder="little").astype(np.float32, copy=False))
    X = np.vstack(vectors)
    cached = read_json(local_path(root, paths["count_cache"]))["records"]
    by_id = {row["record_id"]: {int(k): int(v) for k, v in row["counts"]} for row in cached}
    require(len(by_id) == len(cached) and set(records.record_id) <= set(by_id), "count cache keys incomplete")
    raw_counts = [by_id[rid] for rid in records.record_id]
    positions = {rid: index for index, rid in enumerate(records.record_id)}
    require(set(pairs.record_id_i) | set(pairs.record_id_j) <= set(positions), "pair endpoint missing")
    pi = np.asarray([positions[rid] for rid in pairs.record_id_i], int)
    pj = np.asarray([positions[rid] for rid in pairs.record_id_j], int)
    for pos, field in [(pi, "identity_hash_i"), (pj, "identity_hash_j")]:
        require(np.array_equal(records.iloc[pos].identity_hash.to_numpy(), pairs[field].to_numpy()),
                "pair endpoint chemical identity mismatch")
    for field in ["normalized_poi_id", "pair_context_hash"]:
        require(np.array_equal(records.iloc[pi][field].to_numpy(), records.iloc[pj][field].to_numpy()),
                "pair crosses recorded condition block")
        require(np.array_equal(records.iloc[pi][field].to_numpy(), pairs[field].to_numpy()),
                "pair condition metadata mismatch")
    z = pd.to_numeric(records.pdc50_median).to_numpy(float)
    require(np.isfinite(z).all(), "record label not finite")
    y = z[pj] - z[pi]
    require(np.allclose(y, -pd.to_numeric(pairs.true_delta).to_numpy(float), rtol=0, atol=TOL),
            "canonical direction must equal minus stored source true_delta")
    pairs["true_delta"] = y
    if "is_cliff" in pairs:
        require(np.array_equal(as_bool(pairs.is_cliff), np.abs(y) >= 1 - TOL), "frozen cliff membership differs")
    return Population(records, pairs, X, raw_counts, pi, pj, z)


def read_bundle(bundle):
    spec_path = bundle / "spec.json"
    spec = read_json(spec_path)
    require(spec.get("schema_id") == "R5_PORTABLE_TRAINING_V1", "wrong R5 bundle schema")
    require(spec.get("status") == "FROZEN_BEFORE_FIRST_SCIENTIFIC_FIT", "bundle is not frozen")
    require(set(spec["representations"]) == {"binary", "count"}, "paired representations incomplete")
    require(spec.get("external", {}).get("enabled") is True, "retained exploratory case is not bound")
    freeze_path = local_path(bundle, spec["freeze_receipt"])
    freeze = read_json(freeze_path)
    require(freeze.get("scientific_execution_authorized") is True, "scientific decision missing")
    run_root = local_path(bundle, spec["output_root"])
    binding = read_json(run_root / "RUN_BINDING.json")
    require(binding == {"spec_sha256": sha256(spec_path), "freeze_sha256": sha256(freeze_path),
                        "decision_reference": freeze["decision_reference"]}, "training output belongs to another frozen run")
    # The training gate owns the full frozen manifest. This analysis verifies its
    # immutable run binding and exact completed job/prediction closure only.
    main = load_population(bundle, spec["inputs"])
    external = load_population(bundle, spec["external"])
    config = read_json(local_path(bundle, spec["scientific_config"]))
    jobs = spec["jobs"]
    columns = ["representation", "learner", "repeat", "outer_fold", "pipeline"]
    scope = pd.DataFrame(jobs)
    require(not scope.duplicated(columns).any(), "duplicate frozen diagnostic jobs")
    require(not (scope.learner.eq("ridge") & scope.pipeline.eq("context_pair_C1")).any(), "Ridge C1 not defined")
    require(set(scope.representation) == {"binary", "count"}, "missing representation jobs")
    required_members = {(r, o) for r in range(3) for o in range(5)}
    for keys, cell in scope.groupby(["representation", "learner", "pipeline"]):
        require(set(zip(cell.repeat, cell.outer_fold)) == required_members, "arm lacks exact frozen 3x5 members")
    bkeys = set(map(tuple, scope.loc[scope.representation.eq("binary"), columns[1:]].to_numpy()))
    ckeys = set(map(tuple, scope.loc[scope.representation.eq("count"), columns[1:]].to_numpy()))
    require(bkeys == ckeys, "binary/count job pairing differs")
    outer = pd.read_csv(local_path(bundle, spec["inputs"]["outer_folds"]), keep_default_na=False)
    require(not outer.duplicated(["repeat", "identity_hash"]).any(), "duplicate outer identity assignment")
    outer_maps = {}
    for repeat in sorted(scope.repeat.unique()):
        rows = outer.loc[outer["repeat"].eq(repeat)]
        mapping = dict(zip(rows.identity_hash, rows.outer_fold))
        require(set(mapping) == set(main.records.identity_hash), "outer map identity scope differs")
        outer_maps[int(repeat)] = mapping
    return spec, config, run_root, main, external, jobs, outer_maps


def prediction_tables(run_root, main, external, jobs, outer_maps):
    rows, erows = [], []
    for job in jobs:
        rep, learner, repeat, fold, arm = (job[k] for k in ["representation", "learner", "repeat", "outer_fold", "pipeline"])
        cell = run_root / rep / "results" / f"{learner}_R{repeat}" / f"outer_{fold}" / arm
        require((cell / "final_fit.json").is_file(), f"final fit receipt missing: {cell}")
        for filename, population, collection in [("outer_predictions.csv", main, rows),
                                                  ("external_predictions.csv", external, erows)]:
            path = cell / filename
            require(path.is_file(), f"completed prediction file missing: {path}")
            frame = pd.read_csv(path, keep_default_na=False)
            require(frame.pair_id.is_unique and len(frame) > 0, "duplicate or empty prediction support")
            for key in ["learner", "repeat", "outer_fold", "pipeline"]:
                require(frame[key].eq(job[key]).all(), f"prediction job metadata differs: {key}")
            if filename == "outer_predictions.csv":
                om = outer_maps[repeat]
                mask = population.pairs.identity_hash_i.map(om).eq(fold) & population.pairs.identity_hash_j.map(om).eq(fold)
                expected = population.pairs.loc[mask]
            else:
                expected = population.pairs
            require(set(frame.pair_id) == set(expected.pair_id), "prediction edge support differs from fixed map")
            expected = expected.set_index("pair_id").reindex(frame.pair_id)
            for field in ["record_id_i", "record_id_j", "normalized_poi_id", "pair_context_hash"]:
                require(np.array_equal(frame[field].to_numpy(), expected[field].to_numpy()), "prediction endpoint/block drift")
            require(np.allclose(frame.true_delta.to_numpy(float), expected.true_delta.to_numpy(float), rtol=0, atol=TOL),
                    "prediction canonical truth differs")
            numeric = frame[["raw_prediction", "calibrated_prediction", "scale_c"]].to_numpy(float)
            require(np.isfinite(numeric).all() and frame.scale_c.ge(0).all(), "nonfinite prediction or invalid frozen scale")
            require(np.allclose(frame.raw_prediction * frame.scale_c, frame.calibrated_prediction, rtol=TOL, atol=TOL),
                    "member calibration differs from frozen no-offset c")
            frame.insert(0, "representation", rep)
            collection.append(frame)
    return pd.concat(rows, ignore_index=True), pd.concat(erows, ignore_index=True)


def external_ensemble(members):
    rows = []
    for keys, cell in members.groupby(["representation", "learner", "pipeline", "pair_id"], sort=True):
        require(len(cell) == 15 and not cell.duplicated(["repeat", "outer_fold"]).any(), "external 15-member closure fails")
        metadata = ["record_id_i", "record_id_j", "normalized_poi_id", "pair_context_hash", "true_delta"]
        require(cell[metadata].nunique().eq(1).all(), "external member population drifts")
        rows.append({**dict(zip(["representation", "learner", "pipeline", "pair_id"], keys)),
                     **cell.iloc[0][metadata].to_dict(), "raw_prediction": float(cell.raw_prediction.mean()),
                     "calibrated_prediction": float(cell.calibrated_prediction.mean()), "member_count": len(cell),
                     "repeat": -1, "outer_fold": -1,
                     "ensemble_rule": "equal member edge weights; each member uses its frozen inner c"})
    return pd.DataFrame(rows)


def consistency_tables(predictions, prefix):
    group_keys = ["representation", "repeat", "outer_fold", "learner", "pipeline", "normalized_poi_id", "pair_context_hash"]
    edges, blocks, potentials = [], [], []
    for keys, group in predictions.groupby(group_keys, sort=True, dropna=False):
        values = dict(zip(group_keys, keys))
        frame, phi, summary = project_block(group)
        edges.append(frame)
        blocks.append({**values, **summary})
        potentials.extend({**values, **row} for row in phi)
    edge_table = pd.concat(edges, ignore_index=True)
    aggregation = ["representation", "repeat", "learner", "pipeline"] if prefix == "main" else ["representation", "learner", "pipeline"]
    summaries = []
    for keys, group in edge_table.groupby(aggregation, sort=True):
        row = {**dict(zip(aggregation, keys)), "edges": len(group), "scenario": "batch graph consistency"}
        y = group.true_delta.to_numpy(float)
        for direction in ["raw", "calibrated"]:
            p = group[f"{direction}_prediction"].to_numpy(float)
            q = group[f"{direction}_projected_prediction"].to_numpy(float)
            numerator, denominator = float(np.dot(p-q, p-q)), float(np.dot(p, p))
            row.update({f"{direction}_kappa": float(np.sqrt(numerator/denominator)) if denominator else 0.0,
                        f"{direction}_zero_denominator": denominator == 0,
                        f"{direction}_mae": float(np.mean(np.abs(y-p))),
                        f"{direction}_projected_mae": float(np.mean(np.abs(y-q))),
                        f"{direction}_rmse": float(np.sqrt(np.mean((y-p)**2))),
                        f"{direction}_projected_rmse": float(np.sqrt(np.mean((y-q)**2)))})
            row.update({f"{direction}_{k}": v for k, v in direction_metrics(y, p).items()})
            row.update({f"{direction}_projected_{k}": v for k, v in direction_metrics(y, q).items()})
        summaries.append(row)
    table = pd.DataFrame(summaries)
    outputs = {f"{prefix}_consistency_edges.csv": edge_table, f"{prefix}_consistency_blocks.csv": pd.DataFrame(blocks),
               f"{prefix}_projected_potentials.csv": pd.DataFrame(potentials)}
    if prefix == "main":
        outputs["main_consistency_repeat.csv"] = table
        numeric = [c for c in table.select_dtypes(include=[np.number]) if c != "repeat"]
        outputs["main_consistency_learner_mean.csv"] = table.groupby(["representation", "learner", "pipeline"], sort=True)[numeric].mean().reset_index()
    else:
        outputs["external_consistency.csv"] = table
    return outputs


def training_stability(run_root, jobs):
    rows = []
    expected_final = []
    for job in jobs:
        if job["pipeline"] != "endpoint_E1" or job["learner"] == "ridge":
            continue
        base = run_root / job["representation"] / "results" / f"{job['learner']}_R{job['repeat']}" / f"outer_{job['outer_fold']}" / "endpoint_E1"
        expected_final.append(base / "final/iteration_diagnostics.json")
        require(expected_final[-1].is_file(), "tree E1 final training iteration log missing")
        selected = read_json(base / "selection.json")["selected"]["candidate_id"]
        for path in sorted(base.glob("**/iteration_diagnostics.json")):
            relative = path.relative_to(base).parts
            final = relative[0] == "final"
            require(final or (len(relative) == 4 and relative[0] == "candidates"), "unexpected E1 diagnostic path")
            phase = "outer_final" if final else "inner"
            candidate = selected if final else relative[1]
            inner = -1 if final else int(relative[2].split("_")[1])
            value = read_json(path)
            require(value.get("training_only") is True and value.get("fixed_iterations") == 5, "E1 diagnostic provenance differs")
            require([r["iteration"] for r in value["rounds"]] == list(range(1, 6)), "E1 five fixed iterations incomplete")
            rows.extend({"representation": job["representation"], "learner": job["learner"], "repeat": job["repeat"],
                         "outer_fold": job["outer_fold"], "phase": phase, "candidate_id": candidate, "inner_fold": inner,
                         "diagnostic_path": path.relative_to(run_root).as_posix(), "final_h_shift": value["final_h_shift"], **r}
                        for r in value["rounds"])
    table = pd.DataFrame(rows).sort_values(["representation", "learner", "repeat", "outer_fold", "phase", "candidate_id", "inner_fold", "iteration"])
    final = table.loc[table.phase.eq("outer_final")].copy()
    require(len(final) == len(expected_final) * 5, "final E1 training round closure fails")
    fifth = final.loc[final.iteration.eq(5)]
    metrics = ["training_h_change_rmse", "alpha_change_record_weighted_rmse"]
    summary = fifth.groupby(["representation", "learner"])[metrics].agg(["median", "min", "max"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary["final_models"] = fifth.groupby(["representation", "learner"]).size().to_numpy()
    return {"E1_training_iteration_stability.csv": table, "E1_final_training_iteration_stability.csv": final,
            "E1_final_iteration5_summary.csv": summary}


def raw_record_table(population):
    rows = []
    for record, vector, counts in zip(population.records.itertuples(index=False), population.X_record, population.raw_counts):
        # Same serialized fixed hexadecimal signature as the original raw census.
        fp_hex = format(sum(1 << int(bit) for bit in np.flatnonzero(vector)), "0512X")
        rows.append({"record_id": record.record_id, "identity_hash": record.identity_hash,
                     "binary_2048_signature": hashlib.sha256(fp_hex.encode()).hexdigest().upper(),
                     "raw_sparse_count_signature": signature(counts), "distinct_environment_ids": len(counts),
                     "total_environment_count": sum(counts.values())})
    return pd.DataFrame(rows)


def raw_pair_table(population, record_table):
    lookup = record_table.set_index("record_id")
    a = lookup.reindex(population.pairs.record_id_i)
    b = lookup.reindex(population.pairs.record_id_j)
    table = population.pairs[["pair_id", "record_id_i", "record_id_j"]].copy()
    table["same_chemical_identity"] = a.identity_hash.to_numpy() == b.identity_hash.to_numpy()
    table["binary_zero_difference"] = a.binary_2048_signature.to_numpy() == b.binary_2048_signature.to_numpy()
    table["raw_count_zero_difference"] = a.raw_sparse_count_signature.to_numpy() == b.raw_sparse_count_signature.to_numpy()
    table["true_delta"] = population.pairs.true_delta.to_numpy(float)
    table["binary_forced_zero_abs_error"] = np.abs(table.true_delta) * table.binary_zero_difference
    table["raw_count_forced_zero_abs_error"] = np.abs(table.true_delta) * table.raw_count_zero_difference
    return table


def effective_summary(table, *, repeat, fold, layer, population, scope):
    values = classes_summary(table, "effective_count_signature")
    values["effective_representation_classes"] = values.pop("representation_classes")
    raw_total = float(table.raw_environment_count.sum())
    return {"representation": "count", "repeat": repeat, "outer_fold": fold, "layer": layer,
            "population": population, "scope": scope, **values,
            "effective_zero_vectors": int(table.effective_zero_vector.sum()),
            "vocabulary_size": int(table.vocabulary_size.iloc[0]) if len(table) else None,
            "unknown_id_fraction_mean": float(table.unknown_id_fraction.mean()),
            "unknown_id_fraction_max": float(table.unknown_id_fraction.max()),
            "unknown_count_fraction_mean": float(table.unknown_count_fraction.mean()),
            "unknown_count_fraction_max": float(table.unknown_count_fraction.max()),
            "unknown_count_fraction_pooled": float(table.unknown_environment_count.sum()/raw_total) if raw_total else None}


def representation_tables(bundle, spec, main, external, outer_maps, members, ensemble):
    outputs, raw_classes, class_summaries, vocabulary_rows, census_rows, bounds, external_maps = {}, [], [], [], [], [], []
    raw_main, raw_external = raw_record_table(main), raw_record_table(external)
    for name, data, raw in [("main", main, raw_main), ("external", external, raw_external)]:
        outputs[f"raw_{name}_records.csv"] = raw
        outputs[f"raw_{name}_pair_diagnostics.csv"] = raw_pair_table(data, raw)
        for rep, field in [("binary", "binary_2048_signature"), ("count", "raw_sparse_count_signature")]:
            raw_classes.append({"population_representation": f"{name}_{rep}", **classes_summary(raw, field)})
    binary_maps = {name: pair_inputs(data, data.X_record, "binary_2048", "PREDEFINED_2048", name)
                   for name, data in [("main", main), ("external", external)]}
    for name, table in binary_maps.items():
        outputs[f"pair_inputs/binary/{name}.csv"] = table
    internal_maps = {}
    for repeat, om in outer_maps.items():
        for fold in range(5):
            layer = f"R{repeat}_O{fold}_I-1"
            path = local_path(bundle, spec["inputs"]["vocab_dir"]) / f"{layer}.npz"
            with np.load(path, allow_pickle=False) as value:
                vocab = value["environment_ids"]
                training_ids = value["training_record_ids"].astype(str)
            train = main.records.identity_hash.map(om).ne(fold).to_numpy(bool)
            validation = main.records.identity_hash.map(om).eq(fold).to_numpy(bool)
            ptest = (main.pairs.identity_hash_i.map(om).eq(fold) & main.pairs.identity_hash_j.map(om).eq(fold)).to_numpy(bool)
            ptrain = (main.pairs.identity_hash_i.map(om).ne(fold) & main.pairs.identity_hash_j.map(om).ne(fold)).to_numpy(bool)
            require(ptest.any() and ptrain.any(), "frozen outer layer lacks test or training edges")
            require(np.array_equal(training_ids, main.records.loc[train].record_id.to_numpy(str)), "actual frozen vocabulary training record order differs")
            X, unknown = map_counts(main.raw_counts, vocab)
            E, eunknown = map_counts(external.raw_counts, vocab)
            require(unknown.loc[train, "unknown_environment_ids"].eq(0).all(), "training vocabulary loses its own environment ids")
            mclass = class_table(main.records, X, unknown, layer=layer, population="main_all_eligible", train=train)
            mclass["in_this_layer_validation_scope"] = validation
            eclass = class_table(external.records, E, eunknown, layer=layer, population="external_fixed_seven")
            outputs[f"effective_main/{layer}.csv"], outputs[f"effective_external/{layer}.csv"] = mclass, eclass
            for population, table, scopes in [("main", mclass, {"all": np.ones(len(mclass), bool), "vocabulary_train": train, "outer_test": validation}),
                                               ("external", eclass, {"all": np.ones(len(eclass), bool)})]:
                for scope, mask in scopes.items():
                    class_summaries.append(effective_summary(table.loc[mask], repeat=repeat, fold=fold, layer=layer, population=population, scope=scope))
            zero = np.all(X[main.pair_i] == X[main.pair_j], axis=1)
            ezero = np.all(E[external.pair_i] == E[external.pair_j], axis=1)
            y = main.pairs.true_delta.to_numpy(float)
            vocabulary_rows.append({"layer": layer, "repeat": repeat, "outer_fold": fold, "inner_fold": -1,
                "vocabulary_size": len(vocab), "training_records": int(train.sum()),
                "training_identities": int(main.records.loc[train].identity_hash.nunique()), "train_pairs": int(ptrain.sum()),
                "validation_pairs": int(ptest.sum()), "main_effective_classes": int(mclass.effective_count_signature.nunique()),
                "main_cross_identity_class_count": classes_summary(mclass, "effective_count_signature")["classes_with_distinct_chemical_identities"],
                "external_effective_classes": int(eclass.effective_count_signature.nunique()),
                "all_main_zero_difference_pairs": int(zero.sum()), "test_zero_difference_pairs": int(zero[ptest].sum()),
                "test_forced_zero_mae_contribution": float(np.abs(y[ptest & zero]).sum()/ptest.sum()),
                "external_zero_difference_pairs": int(ezero.sum()),
                "external_forced_zero_mae_contribution": float(np.mean(np.abs(external.pairs.true_delta.to_numpy(float))*ezero))})
            for rep, zero_values in [("binary", binary_maps["main"].exact_zero_d.to_numpy(bool)), ("count", zero)]:
                bounds.append({"representation": rep, "repeat": repeat, "outer_fold": fold, "test_pairs": int(ptest.sum()),
                               "test_zero_difference_pairs": int(zero_values[ptest].sum()),
                               "test_forced_zero_mae_contribution": float(np.abs(y[ptest & zero_values]).sum()/ptest.sum())})
            for population, data, matrix in [("main", main, X), ("external", external, E)]:
                count_table = pair_inputs(data, matrix, "sparse_count", layer, population)
                outputs[f"pair_inputs/count/{layer}/{population}.csv"] = count_table
                for rep, table, source in [("binary", binary_maps[population], f"pair_inputs/binary/{population}.csv"),
                                           ("count", count_table, f"pair_inputs/count/{layer}/{population}.csv")]:
                    census_rows.extend(pair_input_census(table, member_layer=layer, repeat=repeat, outer_fold=fold, source_path=Path(source)))
                    if population == "external":
                        mapped = table.copy()
                        mapped.insert(0, "source_representation", mapped.representation)
                        mapped["representation"], mapped["repeat"], mapped["outer_fold"], mapped["member_layer"] = rep, repeat, fold, layer
                        external_maps.append(mapped)
                if population == "main":
                    internal_maps[(repeat, fold)] = count_table
    outputs["raw_representation_classes.csv"] = pd.DataFrame(raw_classes)
    outputs["effective_input_scope_summary.csv"] = pd.DataFrame(class_summaries)
    outputs["outer_vocabulary_census.csv"] = pd.DataFrame(vocabulary_rows)
    outputs["outer_effective_forced_zero_bounds.csv"] = pd.DataFrame(bounds)
    pcensus = pd.DataFrame(census_rows)
    pcensus.insert(0, "source_representation", pcensus.representation)
    pcensus["representation"] = pcensus.representation.map({"binary_2048": "binary", "sparse_count": "count"})
    outputs["pair_input_class_census.csv"] = pcensus
    emaps = pd.concat(external_maps, ignore_index=True)
    outputs["external_actual_pair_input_members.csv"] = emaps
    member_inputs = emaps[["representation", "repeat", "outer_fold", "pair_id", "record_id_i", "record_id_j", "member_layer", "exact_zero_d"]].rename(
        columns={"exact_zero_d": "effective_zero_difference", "member_layer": "layer"})
    common = member_inputs.groupby(["representation", "pair_id", "record_id_i", "record_id_j"], sort=True).agg(
        zero_difference_members=("effective_zero_difference", "sum"), member_count=("effective_zero_difference", "size")).reset_index()
    require(common.member_count.eq(15).all(), "effective input member count differs")
    common["all_15_members_zero_difference"] = common.zero_difference_members.eq(common.member_count)
    outputs["external_effective_member_inputs.csv"], outputs["external_effective_common_inputs.csv"] = member_inputs, common
    merged = members.merge(member_inputs[["representation", "repeat", "outer_fold", "pair_id", "effective_zero_difference"]],
                           on=["representation"] + KEY, validate="m:1")
    for prefix in ["raw", "calibrated"]:
        merged[f"{prefix}_zero_prediction"] = merged[f"{prefix}_prediction"].abs().le(TOL)
        require(merged.loc[merged.effective_zero_difference, f"{prefix}_zero_prediction"].all(), "same effective input violates antisymmetric zero")
    outputs["external_member_input_and_prediction.csv"] = merged
    ens = ensemble.merge(common, on=["representation", "pair_id", "record_id_i", "record_id_j"], validate="m:1", suffixes=("", "_input"))
    for prefix in ["raw", "calibrated"]:
        ens[f"{prefix}_zero_prediction"] = ens[f"{prefix}_prediction"].abs().le(TOL)
        ens[f"{prefix}_correct_direction"] = np.sign(ens[f"{prefix}_prediction"]).eq(np.sign(ens.true_delta)) & ~ens[f"{prefix}_zero_prediction"]
    outputs["external_ensemble_input_and_prediction.csv"] = ens
    pair_predictions = members.loc[members.pipeline.isin(["direct_pair", "context_pair_C1"])].merge(
        emaps.drop(columns=["record_id_i", "record_id_j"]), on=["representation"] + KEY, validate="m:1")
    is_c1 = pair_predictions.pipeline.eq("context_pair_C1")
    for target, c1, direct in [("actual_pair_input_family", "[d,s]", "d"),
                               ("actual_input_signature", "C1_ds_signature", "direct_d_signature"),
                               ("actual_signed_orbit_id", "C1_signed_orbit_id", "direct_signed_orbit_id"),
                               ("actual_orbit_orientation", "C1_orbit_orientation", "direct_orbit_orientation")]:
        pair_predictions[target] = np.where(is_c1, c1 if target == "actual_pair_input_family" else pair_predictions[c1],
                                           direct if target == "actual_pair_input_family" else pair_predictions[direct])
    outputs["external_member_actual_pair_input_and_prediction.csv"] = pair_predictions
    return outputs, binary_maps, internal_maps, emaps


def output_value_diagnostics(predictions):
    rows = []
    keys = ["representation", "learner", "pipeline", "repeat", "outer_fold"]
    for values, frame in predictions.groupby(keys, sort=True):
        row = {**dict(zip(keys, values)), "external_pairs": len(frame)}
        for prefix in ["raw", "calibrated"]:
            values = frame[f"{prefix}_prediction"]
            frequencies = values.value_counts(dropna=False)
            repeated = frequencies.loc[frequencies.gt(1)]
            row.update({f"{prefix}_zero_predictions": int(values.abs().le(TOL).sum()),
                        f"{prefix}_distinct_exact_predictions": len(frequencies),
                        f"{prefix}_repeated_exact_value_classes": len(repeated),
                        f"{prefix}_pairs_in_repeated_exact_value_classes": int(repeated.sum())})
            anchors, counts = [], []
            for value in sorted(values.to_numpy(float)):
                if not anchors or value-anchors[-1] > TOL:
                    anchors.append(value)
                    counts.append(1)
                else:
                    counts[-1] += 1
            row.update({f"{prefix}_distinct_prediction_classes_tol": len(anchors),
                        f"{prefix}_repeated_prediction_classes_tol": int(np.sum(np.asarray(counts) > 1)),
                        f"{prefix}_pairs_in_repeated_prediction_classes_tol": int(np.sum(np.asarray(counts)[np.asarray(counts) > 1]))})
        rows.append(row)
    return pd.DataFrame(rows)


def observation_map(support, binary, count_maps):
    rows = []
    btable = binary.set_index("pair_id")
    for (repeat, fold), cell in support.groupby(["repeat", "outer_fold"], sort=True):
        ctable = count_maps[(repeat, fold)].set_index("pair_id")
        b, c = btable.reindex(cell.pair_id), ctable.reindex(cell.pair_id)
        require(b.exact_zero_d.notna().all() and c.exact_zero_d.notna().all(), "posthoc input mapping incomplete")
        row = cell.copy()
        row["binary_layer"], row["count_layer"] = "PREDEFINED_2048", f"R{repeat}_O{fold}_I-1"
        row["binary_exact_zero_d"], row["count_exact_zero_d"] = b.exact_zero_d.to_numpy(bool), c.exact_zero_d.to_numpy(bool)
        for rep, table in [("binary", b), ("count", c)]:
            for column in ["direct_d_signature", "direct_signed_orbit_id", "C1_ds_signature", "C1_signed_orbit_id"]:
                row[f"{rep}_{column}"] = table[column].to_numpy()
        row["group"] = classify(row.binary_exact_zero_d, row.count_exact_zero_d)
        row["analysis_status"] = POSTHOC
        rows.append(row)
    table = pd.concat(rows, ignore_index=True)
    require(not table.duplicated(KEY).any(), "posthoc observation mapping duplicated")
    return table


def posthoc_tables(bundle, spec, config, pred, members, ensemble, main, external, binary_maps, count_maps, emaps, output):
    outputs = {}
    support = pred.loc[pred.representation.eq("binary") & pred.learner.eq("ridge") & pred.pipeline.eq("direct_pair")].copy()
    dependency_rel = config["estimands"]["conditional_bootstrap"]["components_source"]
    dependencies = pd.read_csv(local_path(bundle, dependency_rel), keep_default_na=False)
    require(dependencies.pair_id.is_unique, "R5 dependency mapping duplicated")
    support = support.merge(dependencies[["pair_id", "dependency_component"]], on="pair_id", validate="m:1")
    require(len(support) == len(pred.loc[pred.representation.eq("binary") & pred.learner.eq("ridge") & pred.pipeline.eq("direct_pair")]), "dependency support incomplete")
    meta = KEY + ["record_id_i", "record_id_j", "identity_hash_i", "identity_hash_j", "normalized_poi_id", "pair_context_hash", "true_delta", "dependency_component"]
    internal = observation_map(support[meta], binary_maps["main"], count_maps)
    outputs["posthoc_count_only_zero_exceptions_internal.csv"] = internal.loc[internal.group.eq("count_only_zero")]
    if not outputs["posthoc_count_only_zero_exceptions_internal.csv"].empty:
        write_csv(output / "posthoc_count_only_zero_exceptions_internal.csv", outputs["posthoc_count_only_zero_exceptions_internal.csv"])
        raise RuntimeError("count-only-zero present; original three-group closure cannot be used")
    outputs["posthoc_internal_observation_groups.csv"] = internal
    outputs["posthoc_internal_group_census.csv"] = census_tables(internal, "internal_test")
    ipred = attach(pred, internal, "internal")
    fold, fcheck = grouped_metrics(ipred, "internal_test", ["repeat", "outer_fold"])
    repeat, rcheck = grouped_metrics(ipred, "internal_test", ["repeat"])
    mean = repeat_equal_summary(repeat)
    diff, dcheck = representation_differences(repeat)
    dmean = difference_mean(diff)
    outputs.update({"posthoc_internal_group_fold_metrics.csv": fold, "posthoc_internal_group_repeat_metrics.csv": repeat,
                    "posthoc_internal_group_repeatmean_range.csv": mean, "posthoc_internal_group_repeat_loss_differences.csv": diff,
                    "posthoc_internal_group_repeatmean_loss_differences.csv": dmean,
                    "posthoc_internal_zero_prediction_evidence.csv": zero_evidence(ipred, "internal_test", ["repeat", "outer_fold"])})
    external_meta = external.pairs[["pair_id", "identity_hash_i", "identity_hash_j"]]
    base = members.loc[members.representation.eq("binary") & members.learner.eq("ridge") & members.pipeline.eq("direct_pair")].merge(external_meta, on="pair_id", validate="m:1")
    external_count_maps = {(int(r), int(o)): cell for (r, o), cell in emaps.loc[emaps.representation.eq("count")].groupby(["repeat", "outer_fold"])}
    emeta = [c for c in meta if c != "dependency_component"]
    mapping = observation_map(base[emeta], binary_maps["external"], external_count_maps)
    outputs["posthoc_count_only_zero_exceptions_external.csv"] = mapping.loc[mapping.group.eq("count_only_zero")]
    if not outputs["posthoc_count_only_zero_exceptions_external.csv"].empty:
        write_csv(output / "posthoc_count_only_zero_exceptions_external.csv", outputs["posthoc_count_only_zero_exceptions_external.csv"])
        raise RuntimeError("external count-only-zero present; original three-group closure cannot be used")
    agreement = mapping.groupby("pair_id", sort=True).agg(member_count=("group", "size"), distinct_member_groups=("group", "nunique"),
        binary_zero_members=("binary_exact_zero_d", "sum"), count_zero_members=("count_exact_zero_d", "sum")).reset_index()
    outputs["posthoc_external_member_group_agreement.csv"] = agreement
    require(agreement.member_count.eq(15).all(), "external posthoc mapping lacks all members")
    if not agreement.distinct_member_groups.eq(1).all():
        write_csv(output / "posthoc_external_member_group_agreement.csv", agreement)
        raise RuntimeError("external members disagree on input group; no single ensemble three-way classification")
    common = mapping.drop_duplicates("pair_id").copy()
    common["repeat"], common["outer_fold"] = -1, -1
    outputs.update({"posthoc_external_member_observation_groups.csv": mapping, "posthoc_external_ensemble_observation_groups.csv": common,
                    "posthoc_external_member_group_census.csv": census_tables(mapping, "external_member_input"),
                    "posthoc_external_ensemble_group_census.csv": census_tables(common, "external_ensemble")})
    mpred, epred = attach(members, mapping, "external_member"), attach(ensemble, common, "external_ensemble")
    member, mcheck = grouped_metrics(mpred, "external_member", ["repeat", "outer_fold"])
    mrepeat, mrcheck = grouped_metrics(mpred, "external_member", ["repeat"])
    etable, echeck = grouped_metrics(epred, "external_ensemble", [])
    ediff, edcheck = representation_differences(etable, repeat=False)
    outputs.update({"posthoc_external_group_member_metrics.csv": member, "posthoc_external_group_member_repeat_metrics.csv": mrepeat,
                    "posthoc_external_group_member_repeatmean_range.csv": repeat_equal_summary(mrepeat),
                    "posthoc_external_group_ensemble_metrics.csv": etable, "posthoc_external_group_ensemble_loss_differences.csv": ediff,
                    "posthoc_external_member_zero_prediction_evidence.csv": zero_evidence(mpred, "external_member", ["repeat", "outer_fold"]),
                    "posthoc_external_ensemble_zero_prediction_evidence.csv": zero_evidence(epred, "external_ensemble", [])})
    erows = []
    for keys, cell in mpred.groupby(["representation", "learner", "pipeline", "pair_id"], sort=True):
        rep, learner, arm, pair = keys
        selected = epred.loc[epred.representation.eq(rep) & epred.learner.eq(learner) & epred.pipeline.eq(arm) & epred.pair_id.eq(pair)].iloc[0]
        erows.append({"analysis_status": POSTHOC, **dict(zip(["representation", "learner", "pipeline", "pair_id"], keys)),
            "group": selected.group, "structural_zero_d_all_members": bool(selected.structural_zero_d),
            "raw_ensemble_zero_tol": bool(selected.raw_zero_tol), "calibrated_ensemble_zero_tol": bool(selected.calibrated_zero_tol),
            "raw_zero_members_tol": int(cell.raw_zero_tol.sum()), "calibrated_zero_members_tol": int(cell.calibrated_zero_tol.sum()),
            "scale_c_zero_members": int(cell.scale_c.eq(0).sum()),
            "c0_members_with_nonzero_raw_tol": int((cell.scale_c.eq(0) & ~cell.raw_zero_tol).sum()),
            "nonzero_member_raw_ensemble_cancellation_or_tol": bool(selected.raw_zero_tol and not cell.raw_zero_tol.all()),
            "nonzero_member_cal_ensemble_cancellation_or_tol": bool(selected.calibrated_zero_tol and not cell.calibrated_zero_tol.all())})
    outputs["posthoc_external_ensemble_zero_member_evidence.csv"] = pd.DataFrame(erows)
    for scope, table, column in [("internal", dmean, "mean_difference_contribution_to_overall_mae"),
                                  ("external", ediff, "difference_contribution_to_overall_mae")]:
        compact = table.loc[table.scale.eq("calibrated")].pivot(index=["learner", "pipeline"], columns="group", values=column)
        compact = compact.rename(columns={"all": "overall_count_minus_binary_mae"}).reset_index()
        compact["analysis_status"] = POSTHOC
        outputs[f"posthoc_{scope}_calibrated_loss_contributions.csv"] = compact
    outputs["posthoc_group_metric_arithmetic_closure.csv"] = pd.concat([fcheck, rcheck, mcheck, mrcheck, echeck], ignore_index=True)
    outputs["posthoc_representation_loss_difference_closure.csv"] = pd.concat([dcheck, edcheck], ignore_index=True)
    outputs["posthoc_internal_calibrated_three_group_details.csv"] = mean.loc[mean.scale.eq("calibrated") & mean.group.ne("all")].copy()
    outputs["posthoc_external_calibrated_three_group_details.csv"] = etable.loc[etable.scale.eq("calibrated") & etable.group.ne("all")].copy()
    forced = []
    for repeat_id, cell in internal.groupby("repeat", sort=True):
        for rep in ["binary", "count"]:
            mask = cell[f"{rep}_exact_zero_d"]
            loss = float(cell.loc[mask, "true_delta"].abs().sum())
            forced.append({"analysis_status": POSTHOC, "representation": rep, "repeat": int(repeat_id), "repeat_observations": len(cell),
                           "forced_zero_observations": int(mask.sum()), "forced_zero_absolute_loss_sum": loss,
                           "forced_zero_mae_contribution": loss/len(cell)})
    outputs["posthoc_internal_actual_structural_zero_contribution.csv"] = pd.DataFrame(forced)
    return outputs


def write_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g", na_rep="NA")


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def derive(bundle, output):
    spec, config, run_root, main, external, jobs, outer_maps = read_bundle(bundle)
    require(output != bundle and not output.is_relative_to(run_root), "diagnostic output would overwrite training root")
    for section in [spec["inputs"], spec["external"]]:
        for key in ["records", "pairs", "fingerprints", "count_cache"]:
            require(not local_path(bundle, section[key]).is_relative_to(output), "diagnostic output contains frozen inputs")
    pred, members = prediction_tables(run_root, main, external, jobs, outer_maps)
    ensemble = external_ensemble(members)
    outputs = {"main_outer_predictions.csv": pred, "external_per_model_predictions.csv": members,
               "external_ensemble_predictions.csv": ensemble}
    outputs.update(consistency_tables(pred, "main"))
    outputs.update(consistency_tables(ensemble, "external"))
    potentials = outputs["external_projected_potentials.csv"]
    observed = dict(zip(external.records.record_id, external.z - external.z.mean()))
    potentials["observed_centered_pdc50"] = potentials.record_id.map(observed)
    potentials["potential_note"] = "projected batch edge predictions; no per-entity error bars inferred"
    outputs.update(training_stability(run_root, jobs))
    representation, binary_maps, count_maps, emaps = representation_tables(bundle, spec, main, external, outer_maps, members, ensemble)
    outputs.update(representation)
    outputs["external_member_prediction_diagnostics.csv"] = output_value_diagnostics(members)
    outputs["external_ensemble_prediction_diagnostics.csv"] = output_value_diagnostics(ensemble)
    outputs.update(posthoc_tables(bundle, spec, config, pred, members, ensemble, main, external, binary_maps, count_maps, emaps, output))
    manifest = []
    for name, table in outputs.items():
        write_csv(output / name, table)
        manifest.append({"file": name, "rows": len(table), "columns": list(table.columns)})
    receipt = {"status": "COMPLETE_EXISTING_R5_DIAGNOSTICS_NOT_INDEPENDENT_ACCEPTANCE", "run_id": spec.get("run_id"),
        "bundle": bundle.as_posix(), "run_output": spec["output_root"], "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_fits_added": 0, "new_model_predictions": 0, "models_loaded": 0, "new_bootstrap_draws_intervals_tests_sources": 0,
        "main_records": len(main.records), "main_identities": int(main.records.identity_hash.nunique()), "main_pairs": len(main.pairs),
        "external_records": len(external.records), "external_pairs": len(external.pairs), "frozen_final_slots": len(jobs),
        "external_member_rows": len(members), "external_ensemble_rows": len(ensemble), "zero_tolerance": TOL,
        "kappa_definition": "L2 cycle residual / L2 prediction; denominator zero gives zero; record incidence per same repeat/fold/block",
        "projection_labels": "not used for projection; labels enter subsequent error description", "projection_MAE_guarantee": False,
        "posthoc_status": POSTHOC, "groups": GROUPS, "group_equality": "exact actual binary d and member-vocabulary count d",
        "posthoc_weighting": "group loss sum / repetition total observations; pool folds then equally average repetitions",
        "empty_group": "NA group metrics; zero absolute loss sum and zero contribution when full scope nonempty",
        "external_scope": "previously examined exploratory case; member predictions averaged before ensemble loss; no new external source admission",
        "prediction_value_classes": "exact serialized floats plus greedy sorted classes with max group span <= 1e-12; no chaining",
        "input_classes": "record X equality differs from Direct d or C1 [d,s] equality; +0/-0 canonicalized",
        "closure_scope": "run binding, exact frozen member/edge support, canonical truth/calibration, forced zero and posthoc arithmetic only",
        "full_freeze_manifest": "owned by training gate and final acceptance; not duplicated here", "files": manifest,
        "acceptance_owner": "root; one consolidated scientific-stage acceptance pending"}
    dump(output / "DIAGNOSTICS_RECEIPT.json", receipt)
    print(json.dumps({k: receipt[k] for k in ["status", "run_id", "main_records", "main_pairs", "frozen_final_slots", "scientific_fits_added", "new_model_predictions"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path, help="portable_training bundle directory containing frozen spec.json")
    parser.add_argument("--output", required=True, type=Path, help="explicit new diagnostic output directory")
    args = parser.parse_args()
    bundle, output = args.bundle.resolve(), args.output.resolve()
    try:
        derive(bundle, output)
    except (RuntimeError, FileNotFoundError, KeyError, ValueError) as exc:
        print(json.dumps({"status": "DIAGNOSTICS_CLOSED_ON_MISSING_OR_CONFLICTING_INPUT", "reason": str(exc)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
