"""Revised B+C1 nested experiment; all writes are confined to this run root.

Only prepare_runtime.py can freeze the plan/data gate. Each fit checkpoint is
written before FIT_COMPLETE; a completed fold is recovered without refitting.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from model_adapters import fit_model, fit_ridge_e1_within, nonnegative_l1_scale, predict_model

ROOT = Path(__file__).resolve().parent / "UNBOUND_RUNTIME"
CONFIG_PATH = ROOT / "config/frozen_config.json"
FREEZE_PATH = ROOT / "runtime/freeze_receipt.json"
READY_PATH = ROOT / "runtime/PLAN_DATA_READY.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True,
                              default=json_default) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def atomic_model(path: Path, model: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    joblib.dump(model, tmp, compress=1)
    os.replace(tmp, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, path)


def as_bool(series: pd.Series) -> pd.Series:
    tokens = series.astype(str).str.strip().str.lower()
    if not tokens.isin(["true", "false", "1", "0"]).all():
        raise RuntimeError("non-boolean token")
    return tokens.isin(["true", "1"])


def decode_fp(fp_hex: str) -> np.ndarray:
    if not isinstance(fp_hex, str) or len(fp_hex) != 512 or fp_hex != fp_hex.upper():
        raise RuntimeError("invalid fingerprint hex")
    return np.unpackbits(np.frombuffer(bytes.fromhex(fp_hex)[::-1], dtype=np.uint8),
                         bitorder="little").astype(np.float32, copy=False)


def block_id(frame: pd.DataFrame) -> np.ndarray:
    if frame[["normalized_poi_id", "pair_context_hash"]].isna().any().any():
        raise RuntimeError("missing block key")
    return (frame.normalized_poi_id.astype(str) + "\x1f" +
            frame.pair_context_hash.astype(str)).to_numpy(dtype=object)


@dataclass
class DataBundle:
    records: pd.DataFrame
    pairs: pd.DataFrame
    X_record: np.ndarray
    z_record: np.ndarray
    X_diff: np.ndarray
    X_sum: np.ndarray
    y_pair: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray


def load_bundle(cfg: dict[str, Any], external: bool = False) -> DataBundle:
    source = cfg["external_inputs"] if external else cfg["inputs"]
    tables = {}
    for key in ("records", "pairs", "fingerprints"):
        binding = source[key]
        path = Path(binding["path"])
        if sha256(path) != binding["sha256"]:
            raise RuntimeError(f"input SHA mismatch: {key}, external={external}")
        tables[key] = pd.read_csv(path)
    records = tables["records"]
    if "primary_pair_eligible" in records:
        records = records.loc[as_bool(records.primary_pair_eligible)].copy()
    records = records.sort_values("record_id", kind="mergesort").reset_index(drop=True)
    pairs = tables["pairs"].sort_values("pair_id", kind="mergesort").reset_index(drop=True)
    expected = cfg["external_data"] if external else cfg["data"]
    if len(records) != expected["records_eligible"] or len(pairs) != expected["compatible_pairs"]:
        raise RuntimeError("frozen data census mismatch")
    if records.record_id.duplicated().any() or pairs.pair_id.duplicated().any():
        raise RuntimeError("duplicate record/pair key")
    fps = tables["fingerprints"]
    if fps.record_id.duplicated().any():
        raise RuntimeError("duplicate fingerprint key")
    fp_map = fps.set_index("record_id").fp_hex
    if not set(records.record_id) <= set(fp_map.index):
        raise RuntimeError("missing eligible fingerprint")
    X = np.vstack([decode_fp(fp_map.at[rid]) for rid in records.record_id])
    z = pd.to_numeric(records.pdc50_median).to_numpy(float)
    if X.shape != (len(records), 2048) or not np.isfinite(z).all():
        raise RuntimeError("record feature/target failure")
    index = {rid: n for n, rid in enumerate(records.record_id)}
    try:
        pi = np.asarray([index[rid] for rid in pairs.record_id_i], dtype=np.int64)
        pj = np.asarray([index[rid] for rid in pairs.record_id_j], dtype=np.int64)
    except KeyError as exc:
        raise RuntimeError("pair endpoint outside eligible set") from exc
    for indices, column in ((pi, "identity_hash_i"), (pj, "identity_hash_j")):
        if not np.array_equal(records.iloc[indices].identity_hash.to_numpy(), pairs[column].to_numpy()):
            raise RuntimeError(f"endpoint identity mismatch: {column}")
    for col in ("normalized_poi_id", "pair_context_hash"):
        if not np.array_equal(records.iloc[pi][col].to_numpy(), records.iloc[pj][col].to_numpy()):
            raise RuntimeError(f"endpoint block mismatch: {col}")
    y = z[pj] - z[pi]
    if np.max(np.abs(y + pd.to_numeric(pairs.true_delta).to_numpy(float))) > 1e-12:
        raise RuntimeError("canonical direction mismatch: y must equal -source true_delta")
    cliff = as_bool(pairs.is_cliff).to_numpy()
    if not np.array_equal(cliff, np.abs(y) >= 1.0 - 1e-12):
        raise RuntimeError("secondary threshold mismatch")
    pairs["canonical_y"] = y
    pairs["secondary_abs_ge_1"] = cliff
    pairs["pair_context_hash"] = records.iloc[pi].pair_context_hash.to_numpy()
    return DataBundle(records, pairs, X, z, X[pj] - X[pi], X[pj] + X[pi], y, pi, pj)


def candidate_grid(cfg: dict[str, Any], learner: str) -> list[dict[str, Any]]:
    if learner == "ridge":
        rows = [{"candidate_id": f"ridge_{i:02d}", "params": {"alpha": float(a)}}
                for i, a in enumerate(cfg["learners"][learner]["alphas"])]
    else:
        spec = cfg["learners"][learner]
        # Preserve historical candidate_id meanings despite sorted JSON keys.
        names = (["max_depth", "min_samples_leaf", "max_features"] if learner == "random_forest"
                 else ["max_depth", "min_child_weight", "reg_lambda"])
        prefix = "rf" if learner == "random_forest" else "xgb"
        rows = [{"candidate_id": f"{prefix}_{i:02d}",
                 "params": {**spec["common"], **dict(zip(names, values))}}
                for i, values in enumerate(itertools.product(*(spec["grid"][n] for n in names)))]
    if len(rows) != 8:
        raise RuntimeError("candidate count must be eight")
    return rows


def predict_pipeline(model: Any, pipeline: str, bundle: DataBundle, indices: np.ndarray) -> np.ndarray:
    if pipeline == "direct_pair":
        d = bundle.X_diff[indices]
        p = 0.5 * (predict_model(model, d) - predict_model(model, -d))
        p[np.all(d == 0, axis=1)] = 0
    elif pipeline == "context_pair_C1":
        d, s = bundle.X_diff[indices], bundle.X_sum[indices]
        p = 0.5 * (predict_model(model, np.hstack([d, s])) -
                   predict_model(model, np.hstack([-d, s])))
        p[np.all(d == 0, axis=1)] = 0
    elif pipeline in ("endpoint_E0", "endpoint_E0_matched", "endpoint_E1"):
        p = predict_model(model, bundle.X_record[bundle.pair_j[indices]]) - \
            predict_model(model, bundle.X_record[bundle.pair_i[indices]])
    else:
        raise RuntimeError(f"unknown pipeline: {pipeline}")
    if not np.isfinite(p).all():
        raise RuntimeError("nonfinite pipeline prediction")
    return p


class FitLedger:
    def __init__(self, path: Path, base: dict[str, Any]):
        self.path, self.base = path, base

    def event(self, event: str, context: Mapping[str, Any], **extra: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({**self.base, **context, "event": event,
                                    "timestamp_utc": now_utc(), **extra},
                                   default=json_default, sort_keys=True) + "\n")
            handle.flush()


def fit_checkpoint(*, path: Path, ledger: FitLedger, context: dict[str, Any],
                   learner: str, params: dict[str, Any], seed: int, X: np.ndarray,
                   y: np.ndarray, ridge_blocks: np.ndarray | None = None,
                   save_prediction: bool = False, retain_model: bool = True) -> tuple[Any | None, np.ndarray | None, dict[str, Any]]:
    done, model_path, pred_path = path.with_suffix(".json"), path.with_suffix(".joblib"), path.with_suffix(".npz")
    if done.exists():
        receipt = json.loads(done.read_text(encoding="utf-8"))
        model = joblib.load(model_path) if retain_model else None
        pred = np.load(pred_path, allow_pickle=False)["training_prediction"] if save_prediction else None
        return model, pred, receipt
    ledger.event("FIT_ATTEMPT", context)
    try:
        if ridge_blocks is not None:
            model, receipt = fit_ridge_e1_within(float(params["alpha"]), X, y, ridge_blocks)
        else:
            model, receipt = fit_model(learner, params, X, y, seed,
                                       "cuda" if learner == "xgboost" else "cpu")
        if learner == "xgboost" and not str(receipt.get("actual_device", "")).startswith("cuda"):
            raise RuntimeError(f"XGBoost CUDA request fell back: {receipt.get('actual_device')}")
        prediction = predict_model(model, X) if save_prediction else None
        if retain_model:
            atomic_model(model_path, model)
        if save_prediction:
            atomic_npz(pred_path, training_prediction=prediction)
        atomic_json(done, receipt)
        ledger.event("FIT_COMPLETE", context, receipt=receipt)
        return model if retain_model else None, prediction, receipt
    except BaseException as exc:
        ledger.event("FIT_FAILURE", context, error_type=type(exc).__name__, error=str(exc))
        raise


def e1_positions(bundle: DataBundle, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    positions = np.flatnonzero(mask)
    blocks = block_id(bundle.records.iloc[positions])
    counts = pd.Series(blocks).value_counts()
    keep = np.asarray([counts[v] >= 2 for v in blocks], bool)
    stats = {"records_before_singleton_filter": len(positions), "records_after_singleton_filter": int(keep.sum()),
             "singleton_blocks_excluded": int((counts == 1).sum()), "retained_blocks": int((counts >= 2).sum())}
    if not keep.any():
        raise RuntimeError("E1 has no nonsingleton training block")
    return positions[keep], blocks[keep], stats


def fit_unit(*, learner: str, pipeline: str, params: dict[str, Any], seed: int,
             bundle: DataBundle, record_train: np.ndarray, pair_train: np.ndarray,
             predict_indices: np.ndarray, unit: Path, ledger: FitLedger,
             context: dict[str, Any], external: DataBundle | None = None) -> tuple[np.ndarray, dict[str, Any], np.ndarray | None]:
    unit.mkdir(parents=True, exist_ok=True)
    done = unit / "unit_complete.json"
    prediction_path = unit / "predictions.npz"
    if done.exists():
        info = json.loads(done.read_text(encoding="utf-8"))
        with np.load(prediction_path, allow_pickle=False) as values:
            if not np.array_equal(values["pair_indices"], predict_indices):
                raise RuntimeError("completed unit prediction population drift")
            ext = values["external_raw"] if "external_raw" in values else None
            return values["raw_prediction"], info, ext
    fit_params = dict(params)
    info: dict[str, Any] = {}
    if pipeline in ("direct_pair", "context_pair_C1"):
        d, y = bundle.X_diff[pair_train], bundle.y_pair[pair_train]
        if pipeline == "direct_pair":
            X = np.vstack([d, -d])
        else:
            if learner == "ridge":
                raise RuntimeError("C1 is defined for RF/XGB only")
            s = bundle.X_sum[pair_train]
            X = np.vstack([np.hstack([d, s]), np.hstack([-d, s])])
        y = np.concatenate([y, -y])
        if learner == "ridge":
            fit_params["fit_intercept"] = False
        model, _, receipt = fit_checkpoint(path=unit / "fit", ledger=ledger,
                                          context={**context, "backfit_iteration": 0},
                                          learner=learner, params=fit_params, seed=seed, X=X, y=y)
        info = {"fit_count": 1, "fit_receipts": [receipt], "training_direction": "both_directions"}
    elif pipeline in ("endpoint_E0", "endpoint_E0_matched"):
        # Vocabulary mapping above retains the full training identity scope.
        positions = (e1_positions(bundle, record_train)[0]
                     if pipeline == "endpoint_E0_matched" else np.flatnonzero(record_train))
        if learner == "ridge":
            fit_params["fit_intercept"] = True
        model, _, receipt = fit_checkpoint(path=unit / "fit", ledger=ledger,
                                          context={**context, "backfit_iteration": 0}, learner=learner,
                                          params=fit_params, seed=seed, X=bundle.X_record[positions], y=bundle.z_record[positions])
        info = {"fit_count": 1, "fit_receipts": [receipt]}
        if pipeline == "endpoint_E0_matched":
            info["matched_support"] = e1_positions(bundle, record_train)[2]
            info["endpoint_fit_records"] = len(positions)
    elif pipeline == "endpoint_E1":
        positions, blocks, stats = e1_positions(bundle, record_train)
        X, z = bundle.X_record[positions], bundle.z_record[positions]
        if learner == "ridge":
            model, _, receipt = fit_checkpoint(path=unit / "fit", ledger=ledger,
                                              context={**context, "backfit_iteration": 0, **stats},
                                              learner=learner, params=fit_params, seed=seed, X=X, y=z, ridge_blocks=blocks)
            info = {"fit_count": 1, "fit_receipts": [receipt], "singleton_filter": stats}
        else:
            keys, codes = np.unique(blocks.astype(str), return_inverse=True)
            counts = np.bincount(codes)
            h = np.zeros(len(z), float)
            previous_alpha = np.zeros(len(keys), float)
            receipts, diagnostics = [], []
            for iteration in range(1, 6):
                raw_alpha = np.bincount(codes, weights=z-h) / counts
                centre = float(np.mean(raw_alpha[codes]))
                alpha = raw_alpha - centre
                target = z - alpha[codes]
                step = unit / f"iteration_{iteration}"
                model, new_h, receipt = fit_checkpoint(path=step, ledger=ledger,
                    context={**context, "backfit_iteration": iteration, **stats},
                    learner=learner, params=fit_params, seed=seed, X=X, y=target,
                    save_prediction=True, retain_model=(iteration == 5))
                diagnostic = {"iteration": iteration, "alpha_center": centre,
                    "training_h_change_rmse": float(np.sqrt(np.mean((new_h-h)**2))),
                    "training_h_change_max_abs": float(np.max(np.abs(new_h-h))),
                    "alpha_change_record_weighted_rmse": float(np.sqrt(np.mean((alpha[codes]-previous_alpha[codes])**2))),
                    "alpha_change_max_abs": float(np.max(np.abs(alpha-previous_alpha))),
                    "training_target_rmse": float(np.sqrt(np.mean((target-new_h)**2)))}
                # Checkpoint vectors describe training only; no heldout truth enters iterations.
                diag_path = unit / f"iteration_{iteration}_diagnostic.npz"
                if not diag_path.exists():
                    atomic_npz(diag_path, record_positions=positions, record_ids=bundle.records.iloc[positions].record_id.to_numpy(str),
                               block_keys=keys, block_codes=codes, alpha=alpha, training_target=target,
                               training_prediction=new_h)
                diagnostics.append(diagnostic)
                receipts.append({**receipt, **diagnostic})
                h, previous_alpha = new_h, alpha
            final_raw_alpha = np.bincount(codes, weights=z-h) / counts
            final_centre = float(np.mean(final_raw_alpha[codes]))
            atomic_npz(unit / "final_alpha.npz", block_keys=keys, alpha=final_raw_alpha-final_centre)
            atomic_json(unit / "iteration_diagnostics.json", {"training_only": True, "fixed_iterations": 5,
                        "rounds": diagnostics, "final_h_shift": final_centre})
            info = {"fit_count": 5, "fit_receipts": receipts, "singleton_filter": stats,
                    "final_h_shift": final_centre, "final_alpha_blocks": len(keys),
                    "iteration_diagnostics": (unit / "iteration_diagnostics.json").as_posix()}
    else:
        raise RuntimeError(f"unknown pipeline: {pipeline}")
    if model is None:
        raise RuntimeError("final model missing from fit checkpoint")
    prediction = predict_pipeline(model, pipeline, bundle, predict_indices)
    ext = predict_pipeline(model, pipeline, external, np.arange(len(external.pairs))) if external is not None else None
    arrays = {"pair_indices": predict_indices, "raw_prediction": prediction}
    if ext is not None:
        arrays["external_raw"] = ext
    atomic_npz(prediction_path, **arrays)
    if context["phase"] in ("outer_final", "source_final"):
        model_output_root = Path(context.get("model_output_root", ROOT / "models"))
        final_model_path = model_output_root / context["batch_id"] / f"outer_{context['outer_fold']}" / f"{pipeline}.joblib"
        atomic_model(final_model_path, model)
        info["model_path"] = final_model_path.as_posix()
    atomic_json(done, {**info, "completed_at_utc": now_utc()})
    # A complete unit no longer needs its temporary model; outer models live in models/.
    for path in unit.glob("*.joblib"):
        path.unlink()
    return prediction, info, ext


def count_ledger_events(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)["event"]
            counts[event] = counts.get(event, 0) + 1
    return counts


def verify_freeze() -> dict[str, Any]:
    raise RuntimeError("Use portable.py with an explicit prepared-input manifest to run this worker")


def run_pipeline(*, cfg: dict[str, Any], learner: str, repeat: int, outer_fold: int,
                 bundle: DataBundle, external: DataBundle | None, outer_map: dict[str, int],
                 inner_map: dict[str, int], pipeline: str, cell_root: Path,
                 ledger: FitLedger, seed: int, update_status: Any) -> dict[str, Any]:
    output = cell_root / pipeline
    output.mkdir(parents=True, exist_ok=True)
    record_outer = bundle.records.identity_hash.map(outer_map).to_numpy()
    pair_outer_i, pair_outer_j = bundle.pairs.identity_hash_i.map(outer_map).to_numpy(), bundle.pairs.identity_hash_j.map(outer_map).to_numpy()
    record_train = record_outer != outer_fold
    pair_train = (pair_outer_i != outer_fold) & (pair_outer_j != outer_fold)
    test_indices = np.flatnonzero((pair_outer_i == outer_fold) & (pair_outer_j == outer_fold))
    if not pair_train.any() or not len(test_indices):
        raise RuntimeError("empty outer train/test")
    summaries, common = [], None
    candidates = candidate_grid(cfg, learner)
    batch_id = f"{learner}_R{repeat}"
    for candidate in candidates:
        cid = candidate["candidate_id"]
        candidate_root = output / "candidates" / cid
        complete = candidate_root / "candidate_complete.json"
        saved_path = candidate_root / "oof_predictions.npz"
        if complete.exists():
            summary = json.loads(complete.read_text(encoding="utf-8"))
            with np.load(saved_path, allow_pickle=False) as values:
                indices, raw = values["pair_indices"], values["raw_prediction"]
        else:
            folds_i, folds_p, fits = [], [], []
            ri = bundle.records.identity_hash.map(inner_map)
            pii, pjj = bundle.pairs.identity_hash_i.map(inner_map), bundle.pairs.identity_hash_j.map(inner_map)
            for inner_fold in range(3):
                rtrain = ri.notna().to_numpy() & ri.ne(inner_fold).to_numpy()
                ptrain = pii.notna().to_numpy() & pjj.notna().to_numpy() & pii.ne(inner_fold).to_numpy() & pjj.ne(inner_fold).to_numpy()
                idx = np.flatnonzero(pii.eq(inner_fold).to_numpy() & pjj.eq(inner_fold).to_numpy())
                if not ptrain.any() or not len(idx):
                    raise RuntimeError("empty inner train/validation")
                context = {"batch_id": batch_id, "outer_fold": outer_fold, "pipeline": pipeline,
                           "phase": "inner", "candidate_id": cid, "inner_fold": inner_fold}
                pred, info, _ = fit_unit(learner=learner, pipeline=pipeline, params=candidate["params"], seed=seed,
                    bundle=bundle, record_train=rtrain, pair_train=ptrain, predict_indices=idx,
                    unit=candidate_root / f"inner_{inner_fold}", ledger=ledger, context=context)
                folds_i.append(idx); folds_p.append(pred)
                fits.append({"inner_fold": inner_fold, "validation_pairs": len(idx),
                             **{k: v for k, v in info.items() if k != "fit_receipts"}})
                update_status({"stage": "inner_unit_complete", **context})
            indices, raw = np.concatenate(folds_i), np.concatenate(folds_p)
            order = np.argsort(indices, kind="mergesort")
            indices, raw = indices[order], raw[order]
            if len(np.unique(indices)) != len(indices):
                raise RuntimeError("duplicate OOF pair")
            c = nonnegative_l1_scale(bundle.y_pair[indices], raw)
            summary = {"candidate_id": cid, "params": candidate["params"], "inner_oof_pairs": len(indices),
                       "scale_c": c, "inner_oof_raw_mae": float(np.mean(np.abs(bundle.y_pair[indices]-raw))),
                       "inner_oof_calibrated_mae": float(np.mean(np.abs(bundle.y_pair[indices]-c*raw))),
                       "fold_fit_info": fits}
            atomic_npz(saved_path, pair_indices=indices, raw_prediction=raw)
            atomic_json(complete, summary)
        if common is None:
            common = indices
        elif not np.array_equal(common, indices):
            raise RuntimeError("candidate OOF populations differ")
        summaries.append(summary)
    best = min(row["inner_oof_calibrated_mae"] for row in summaries)
    selected = min((row for row in summaries if row["inner_oof_calibrated_mae"] <= best+1e-12),
                   key=lambda row: row["candidate_id"])
    atomic_json(output / "selection.json", {"selection_metric": "calibrated inner-OOF pair MAE",
                "tie_tolerance": 1e-12, "selected": selected, "candidates": summaries})
    context = {"batch_id": batch_id, "outer_fold": outer_fold, "pipeline": pipeline,
               "phase": "outer_final", "candidate_id": selected["candidate_id"], "inner_fold": -1}
    raw, info, ext = fit_unit(learner=learner, pipeline=pipeline, params=selected["params"], seed=seed,
        bundle=bundle, record_train=record_train, pair_train=pair_train, predict_indices=test_indices,
        unit=output / "final", ledger=ledger, context=context, external=external)
    c = float(selected["scale_c"])
    test = bundle.pairs.iloc[test_indices]
    train_g = set(bundle.records.loc[record_train, "normalized_poi_id"])
    frame = pd.DataFrame({"repeat": repeat, "outer_fold": outer_fold, "learner": learner, "pipeline": pipeline,
        "model_seed": seed, "pair_row_index": test_indices, "pair_id": test.pair_id.to_numpy(),
        "record_id_i": test.record_id_i.to_numpy(), "record_id_j": test.record_id_j.to_numpy(),
        "identity_hash_i": test.identity_hash_i.to_numpy(), "identity_hash_j": test.identity_hash_j.to_numpy(),
        "normalized_poi_id": test.normalized_poi_id.to_numpy(), "pair_context_hash": test.pair_context_hash.to_numpy(),
        "g_seen_in_outer_train": test.normalized_poi_id.isin(train_g).to_numpy(), "true_delta": bundle.y_pair[test_indices],
        "secondary_abs_ge_1": test.secondary_abs_ge_1.to_numpy(), "raw_prediction": raw,
        "scale_c": c, "calibrated_prediction": c*raw, "selected_candidate_id": selected["candidate_id"]})
    atomic_csv(output / "outer_predictions.csv", frame)
    if external is not None:
        external_frame = external.pairs[["pair_id", "record_id_i", "record_id_j", "normalized_poi_id", "pair_context_hash"]].copy()
        external_frame["true_delta"] = external.y_pair
        external_frame["repeat"], external_frame["outer_fold"] = repeat, outer_fold
        external_frame["learner"], external_frame["pipeline"] = learner, pipeline
        external_frame["raw_prediction"], external_frame["calibrated_prediction"] = ext, c*ext
        external_frame["scale_c"], external_frame["selected_candidate_id"] = c, selected["candidate_id"]
        external_frame["ensemble_weight"] = 1/15
        atomic_csv(output / "external_predictions.csv", external_frame)
    truth = frame.true_delta.to_numpy(float)
    summary = {"pipeline": pipeline, "outer_test_pairs": len(frame), "selected_candidate_id": selected["candidate_id"],
               "scale_c": c, "raw_mae": float(np.mean(np.abs(truth-raw))),
               "calibrated_mae": float(np.mean(np.abs(truth-c*raw))),
               "raw_rmse": float(np.sqrt(np.mean((truth-raw)**2))),
               "calibrated_rmse": float(np.sqrt(np.mean((truth-c*raw)**2))),
               "final_fit": {**info, "outer_train_pairs": int(pair_train.sum()),
                             "outer_train_records_E0": int(record_train.sum())}}
    atomic_json(output / "final_fit.json", summary["final_fit"])
    atomic_json(output / "summary.json", summary)
    update_status({"stage": "pipeline_complete", **context})
    return summary


def run_batch(learner: str, repeat: int) -> None:
    cfg = verify_freeze()
    if repeat not in range(3):
        raise RuntimeError("invalid repeat")
    bundle, external = load_bundle(cfg), load_bundle(cfg, external=True)
    outer_all = pd.read_csv(ROOT / "data/outer_identity_folds.csv")
    inner_all = pd.read_csv(ROOT / "config/inner_identity_folds.csv")
    outer = outer_all.loc[outer_all.repeat.eq(repeat)]
    if outer.identity_hash.duplicated().any() or set(outer.identity_hash) != set(bundle.records.identity_hash):
        raise RuntimeError("outer identity coverage failure")
    outer_map = dict(zip(outer.identity_hash, outer.outer_fold))
    seed = int(cfg["splitting"]["model_seeds_paired_by_repeat"][repeat])
    batch_id = f"{learner}_R{repeat}"
    batch_root = ROOT / "results" / batch_id
    batch_root.mkdir(parents=True, exist_ok=True)
    ledger_path = ROOT / "logs" / batch_id / "fit_ledger.jsonl"
    ledger = FitLedger(ledger_path, {"learner": learner, "repeat": repeat, "model_seed": seed})
    status_path = ROOT / "runtime" / batch_id / "batch_status.json"
    started = time.time()
    def update_status(extra: dict[str, Any]) -> None:
        payload = {"status": "RUNNING", "batch_id": batch_id, "pid": os.getpid(), "cwd": Path.cwd().as_posix(),
                   "started_at_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
                   "updated_at_utc": now_utc(), "elapsed_seconds": time.time()-started,
                   "ledger_events": count_ledger_events(ledger_path), **extra}
        atomic_json(status_path, payload)
    update_status({"stage": "batch_start"})
    try:
        cells = []
        for outer_fold in range(5):
            if (ROOT / "runtime/STOP_REQUESTED").exists():
                atomic_json(status_path, {"status": "STOPPED_BY_REQUEST", "batch_id": batch_id,
                            "next_outer_fold": outer_fold, "updated_at_utc": now_utc()})
                return
            cell = batch_root / f"outer_{outer_fold}"
            complete = cell / "cell_complete.json"
            if complete.exists():
                cells.append(json.loads(complete.read_text(encoding="utf-8")))
                continue
            inner = inner_all.loc[inner_all.repeat.eq(repeat) & inner_all.outer_fold.eq(outer_fold)]
            training_ids = {identity for identity, fold in outer_map.items() if fold != outer_fold}
            if inner.identity_hash.duplicated().any() or set(inner.identity_hash) != training_ids:
                raise RuntimeError("inner assignment is not exact outer-train identity set")
            inner_map = dict(zip(inner.identity_hash, inner.inner_fold))
            pipelines = ["direct_pair", "endpoint_E0", "endpoint_E1"]
            if learner != "ridge":
                pipelines.append("context_pair_C1")
            summaries = [run_pipeline(cfg=cfg, learner=learner, repeat=repeat, outer_fold=outer_fold,
                        bundle=bundle, external=external, outer_map=outer_map, inner_map=inner_map,
                        pipeline=pipeline, cell_root=cell, ledger=ledger, seed=seed, update_status=update_status)
                         for pipeline in pipelines]
            result = {"status": "COMPLETE", "batch_id": batch_id, "outer_fold": outer_fold,
                      "pipelines": summaries, "completed_at_utc": now_utc()}
            atomic_json(complete, result)
            cells.append(result)
            update_status({"stage": "outer_cell_complete", "outer_fold": outer_fold})
        summary = {"status": "COMPLETE", "batch_id": batch_id, "learner": learner, "repeat": repeat,
                   "outer_cells_complete": len(cells), "ledger_events": count_ledger_events(ledger_path),
                   "elapsed_seconds_this_invocation": time.time()-started, "completed_at_utc": now_utc(), "cells": cells}
        expected = 375 if learner == "ridge" else 1000
        if summary["ledger_events"].get("FIT_COMPLETE", 0) != expected:
            raise RuntimeError(f"completed fit count mismatch for {batch_id}: expected {expected}")
        atomic_json(batch_root / "batch_complete.json", summary)
        atomic_json(status_path, summary)
        print(f"PASS_BATCH_COMPLETE {batch_id}", flush=True)
    except BaseException as exc:
        atomic_json(status_path, {"status": "FAILED_RECOVERABLE", "batch_id": batch_id, "pid": os.getpid(),
                    "updated_at_utc": now_utc(), "ledger_events": count_ledger_events(ledger_path),
                    "error_type": type(exc).__name__, "error": str(exc),
                    "resume_command": f"& '{sys.executable}' '{Path(__file__).resolve()}' --learner {learner} --repeat {repeat}"})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learner", required=True, choices=["ridge", "random_forest", "xgboost"])
    parser.add_argument("--repeat", type=int, required=True)
    args = parser.parse_args()
    run_batch(args.learner, args.repeat)
