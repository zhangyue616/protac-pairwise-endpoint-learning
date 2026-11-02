"""objective sensitivity LOCO cache reader and one-candidate outer Ridge adapter.

Importing this module does not read data, load models, fit, or predict.  The
caller owns calibration, selection, resource checks, and the execution ledger.
Historical runners and output directories are never used as execution targets.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


ALPHAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
CANDIDATES = tuple(f"ridge_{index:02d}" for index in range(8))
REPRESENTATIONS = ("binary", "count")
PIPELINES = ("endpoint_E1", "endpoint_E0_matched")


def _require(value, message):
    if not value:
        raise RuntimeError(message)


class LocoBranch:
    """Bind the 42 repaired components plus the accepted DEP_006 caches."""

    def __init__(self, manifest, output_root):
        self.manifest_path = Path(manifest).resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.data_dir = self._resolve(self.manifest["data_dir"])
        self.legacy_pairs = self._resolve(self.manifest["legacy_pair_table"]) if self.manifest.get("legacy_pair_table") else None
        self.output_root = Path(output_root).resolve()
        self.source_paths = set()
        self._slots = None
        self._population = None
        self._old_pairs = None
        self._raw_counts = None
        self._feature_cache = None
        self._adapter = None
        self._fit_attempts = set()

    def _source(self, path):
        path = Path(path).resolve()
        self.source_paths.add(path)
        return path

    def _json(self, path):
        return json.loads(self._source(path).read_text(encoding="utf-8"))

    def _csv(self, path):
        return pd.read_csv(self._source(path), keep_default_na=False)

    def _resolve(self, value):
        path = Path(value)
        return (path if path.is_absolute() else self.manifest_path.parent / path).resolve()

    def slots(self):
        if self._slots is not None:
            return self._slots
        rows = []
        for entry in self.manifest["slots"]:
            group, rep, pipeline = entry["outer_fold"], entry["representation"], entry["pipeline"]
            _require(rep in REPRESENTATIONS and pipeline in PIPELINES, "Unexpected sensitivity arm")
            _require(set(entry["old_scores"]) == set(CANDIDATES), "Incomplete frozen alpha grid")
            _require(entry["old_candidate_id"] in CANDIDATES, "Unknown selected candidate")
            rows.append({**entry, "slot_id": f"loco_{rep}_{group}_{pipeline}",
                         "domain": "loco", "repeat": 0,
                         "_cell": str(self._resolve(entry["cell"])),
                         "_test_path": str(self._resolve(entry["test_predictions"]))})
        keys = {(r["outer_fold"], r["representation"], r["pipeline"]) for r in rows}
        groups = {r["outer_fold"] for r in rows}
        _require(len(groups) == 43 and len(rows) == len(keys) == 172 and keys ==
                 {(g, r, p) for g in groups for r in REPRESENTATIONS for p in PIPELINES},
                 "LOCO must contain the 172 unique frozen slots")
        self._slots = rows
        return rows

    def _data(self):
        if self._population is not None:
            return self._population
        records = self._csv(self.data_dir / "records_revised.csv").sort_values(
            "record_id", kind="mergesort").reset_index(drop=True)
        pairs = self._csv(self.data_dir / "pairs_revised.csv").sort_values(
            "pair_id", kind="mergesort").reset_index(drop=True)
        groups = self._csv(self.data_dir / "record_source_group_map.csv")
        _require(len(records) == 699 and records.record_id.is_unique and
                 len(pairs) == 5804 and pairs.pair_id.is_unique and groups.record_id.is_unique,
                 "LOCO current population differs")
        _require(records.primary_pair_eligible.astype(str).str.lower().eq("true").all(),
                 "LOCO record eligibility differs")
        ix = {rid: i for i, rid in enumerate(records.record_id)}
        ii = np.asarray([ix[rid] for rid in pairs.record_id_i], dtype=np.int64)
        jj = np.asarray([ix[rid] for rid in pairs.record_id_j], dtype=np.int64)
        mapping = groups.set_index("record_id").component_id.to_dict()
        rg = np.asarray([mapping[rid] for rid in records.record_id], dtype=str)
        pg = rg[ii]
        _require(np.array_equal(pg, rg[jj]) and len(set(pg)) == 43,
                 "LOCO endpoint components conflict")
        identities = records.identity_hash.to_numpy(str)
        _require(np.array_equal(identities[ii], pairs.identity_hash_i.to_numpy(str)) and
                 np.array_equal(identities[jj], pairs.identity_hash_j.to_numpy(str)),
                 "LOCO endpoint chemical identities conflict")
        for column in ("normalized_poi_id", "pair_context_hash"):
            values = records[column].to_numpy(str)
            _require(np.array_equal(values[ii], values[jj]), "LOCO pair context conflicts")
        z = records.pdc50_median.to_numpy(np.float64)
        y = z[jj] - z[ii]
        _require(np.isfinite(z).all() and np.isfinite(y).all() and
                 np.allclose(y, pairs.canonical_target_b_minus_a.to_numpy(np.float64),
                             rtol=0, atol=1e-12), "LOCO current pair target direction conflicts")
        blocks = (records.normalized_poi_id.astype(str) + "\x1f" +
                  records.pair_context_hash.astype(str)).to_numpy(object)
        authority = pairs.copy()
        authority["_component"] = pg
        authority["_y"] = y
        self._population = dict(records=records, pairs=pairs, authority=authority.set_index("pair_id"),
                                ii=ii, jj=jj, rg=rg, pg=pg, z=z, y=y, blocks=blocks)
        return self._population

    def _arrays(self, frame, heldout, *, is_test):
        frame = frame.sort_values("pair_id", kind="mergesort").reset_index(drop=True)
        _require(len(frame) > 0 and frame.pair_id.is_unique, "LOCO cache has empty/duplicate pairs")
        population = self._data()
        authority = population["authority"]
        _require(frame.pair_id.isin(authority.index).all(), "LOCO cached pair absent from current authority")
        expected = authority.loc[frame.pair_id]
        for column in ("record_id_i", "record_id_j", "identity_hash_i", "identity_hash_j"):
            if column in frame:
                _require(np.array_equal(frame[column].to_numpy(str), expected[column].to_numpy(str)),
                         "LOCO cached pair endpoint identity or direction differs: " + column)
        y = frame.true_delta.to_numpy(np.float64)
        raw = frame.raw_prediction.to_numpy(np.float64)
        components = expected._component.to_numpy(str)
        _require(np.isfinite(y).all() and np.isfinite(raw).all() and
                 np.allclose(y, expected._y.to_numpy(np.float64), rtol=0, atol=1e-12),
                 "LOCO cached labels/predictions nonfinite or label direction differs")
        if "component_id" in frame:
            _require(np.array_equal(frame.component_id.to_numpy(str), components),
                     "LOCO cached component IDs conflict")
        if is_test:
            expected_ids = population["pairs"].loc[population["pg"] == heldout, "pair_id"].to_numpy(str)
            _require(np.array_equal(frame.pair_id.to_numpy(str), expected_ids),
                     "LOCO test population differs from full heldout component")
        else:
            _require(not np.any(components == heldout) and len(set(components)) == 42,
                     "LOCO OOF includes heldout component or does not cover 42 components")
        return {"pair_ids": frame.pair_id.to_numpy(str), "component_ids": components,
                "y": y, "raw": raw}

    def load_oof(self, slot, cid):
        _require(cid in CANDIDATES, "LOCO candidate outside frozen grid")
        cell = Path(slot["_cell"])
        if slot.get("oof_format", "csv") == "csv":
            frame = self._csv(cell / f"oof/{slot['representation']}/{slot['pipeline']}/{cid}.csv")
        else:
            self._json(cell / f"candidates/{cid}/candidate_complete.json")
            npz_path = self._source(cell / f"candidates/{cid}/oof_predictions.npz")
            with np.load(npz_path, allow_pickle=False) as saved:
                indices = saved["pair_indices"].copy()
                raw = saved["raw_prediction"].copy()
            if self._old_pairs is None:
                self._old_pairs = self._csv(self.legacy_pairs).sort_values(
                    "pair_id", kind="mergesort").reset_index(drop=True)
            _require(np.issubdtype(indices.dtype, np.integer) and indices.ndim == 1 and
                     raw.ndim == 1 and len(indices) == len(raw) == 3582 and
                     len(np.unique(indices)) == len(indices) and
                     np.all((indices >= 0) & (indices < len(self._old_pairs))),
                     "DEP_006 saved pair indices invalid")
            # Indices refer to the historical 5788-pair table, never the current table.
            frame = self._old_pairs.iloc[indices].copy().reset_index(drop=True)
            frame["true_delta"] = frame.canonical_target_b_minus_a.to_numpy(np.float64)
            frame["raw_prediction"] = raw
        return self._arrays(frame, slot["outer_fold"], is_test=False)

    def load_test(self, slot):
        frame = self._csv(slot["_test_path"])
        frame = frame.loc[(frame.learner == "ridge") &
                          (frame.representation == slot["representation"]) &
                          (frame.pipeline == slot["pipeline"])].copy()
        _require(frame.selected_candidate_id.eq(slot["old_candidate_id"]).all() and
                 np.allclose(frame.scale_c.to_numpy(np.float64), slot["old_scale"], rtol=0, atol=1e-12),
                 "LOCO old test selection/scale conflicts")
        return self._arrays(frame, slot["outer_fold"], is_test=True)

    def _features(self, rep, heldout):
        key = (rep, heldout)
        if self._feature_cache is not None and self._feature_cache[0] == key:
            return self._feature_cache[1:]
        data = self._data()
        rids = data["records"].record_id.to_numpy(str)
        if rep == "binary":
            fingerprints = self._csv(self.data_dir / "fingerprints.csv").set_index("record_id").fp_hex.to_dict()
            X = np.stack([np.unpackbits(np.frombuffer(bytes.fromhex(fingerprints[rid])[::-1],
                         dtype=np.uint8), bitorder="little").astype(np.float32) for rid in rids])
            _require(X.shape == (699, 2048), "LOCO binary feature width differs")
            info = {"representation": "binary", "features": 2048}
        else:
            _require(rep == "count", "LOCO unsupported representation")
            if self._raw_counts is None:
                cache = self._json(self.data_dir / "main_raw_sparse_counts.json")
                self._raw_counts = {row["record_id"]: {int(k): int(v) for k, v in row["counts"]}
                                    for row in cache["records"]}
            manifest = self._json(self.data_dir / "vocab_manifest.json")
            layer = f"R0_G{heldout}_I-1"
            item = manifest[layer]
            path = self._source((self.data_dir / item["path"]).resolve())
            _require(hashlib.sha256(path.read_bytes()).hexdigest().upper() == item["sha256"].upper(),
                     "LOCO vocabulary manifest identity differs")
            with np.load(path, allow_pickle=False) as saved:
                environments = saved["environment_ids"].copy()
                training_ids = saved["training_record_ids"].copy()
            train = data["rg"] != heldout
            _require(np.array_equal(training_ids, rids[train]),
                     "LOCO outer vocabulary ordered full-training support differs")
            _require(environments.tolist() == sorted({env for rid in training_ids for env in self._raw_counts[rid]}),
                     "LOCO outer vocabulary is not the full training-only environment union")
            columns = {int(environment): index for index, environment in enumerate(environments)}
            X = np.zeros((len(rids), len(environments)), dtype=np.float32)
            unknown_ids = unknown_count = 0
            for index, rid in enumerate(rids):
                for environment, value in self._raw_counts[rid].items():
                    column = columns.get(environment)
                    if column is None:
                        unknown_ids += 1
                        unknown_count += value
                    else:
                        X[index, column] = value
            info = {"representation": "count", "features": len(environments),
                    "vocabulary_layer": layer, "vocabulary_path": str(path),
                    "vocabulary_training_records": len(training_ids),
                    "unknown_environment_ids_across_records": unknown_ids,
                    "unknown_environment_count_across_records": unknown_count,
                    "unknown_handling": "discard; no external or heldout vocabulary expansion"}
        self._feature_cache = (key, X, info)
        return X, info

    def _adapters(self):
        if self._adapter is None:
            path = self._source(Path(__file__).resolve().parents[1] / "core/model_adapters.py")
            spec = importlib.util.spec_from_file_location("objective_loco_frozen_model_adapters", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._adapter = module
        return self._adapter

    def fit_predict(self, slot, cid, output_dir):
        _require(cid in CANDIDATES, "LOCO requested fit candidate outside frozen alpha grid")
        key = (slot["slot_id"], cid)
        _require(key not in self._fit_attempts, "LOCO duplicate outer fit attempt prohibited")
        destination = Path(output_dir).resolve()
        _require(destination.is_relative_to(self.output_root), "LOCO output escapes objective sensitivity root")
        model_path, receipt_path = destination / "model.joblib", destination / "receipt.json"
        _require(not model_path.exists() and not receipt_path.exists(), "LOCO outer fit output already exists")
        test = self.load_test(slot)
        data = self._data()
        heldout = slot["outer_fold"]
        train = data["rg"] != heldout
        counts = Counter(data["blocks"][train])
        support = train & np.asarray([counts.get(block, 0) >= 2 for block in data["blocks"]])
        _require(support.any(), "LOCO empty matched/E1 endpoint training support")
        X, feature_info = self._features(slot["representation"], heldout)
        adapter = self._adapters()
        alpha = ALPHAS[CANDIDATES.index(cid)]
        destination.mkdir(parents=True, exist_ok=True)
        self._fit_attempts.add(key)
        with threadpool_limits(limits=4):
            if slot["pipeline"] == "endpoint_E1":
                model, receipt = adapter.fit_ridge_e1_within(
                    alpha, X[support], data["z"][support], data["blocks"][support])
            else:
                _require(slot["pipeline"] == "endpoint_E0_matched", "LOCO unsupported pipeline")
                model, receipt = adapter.fit_model("ridge", {"alpha": alpha, "fit_intercept": True},
                                                   X[support], data["z"][support], 20261017, "cpu")
            mask = data["pg"] == heldout
            ii, jj = data["ii"][mask], data["jj"][mask]
            raw = adapter.predict_model(model, X[jj]) - adapter.predict_model(model, X[ii])
        _require(np.array_equal(test["pair_ids"], data["pairs"].loc[mask, "pair_id"].to_numpy(str)) and
                 raw.shape == test["raw"].shape and np.isfinite(raw).all(),
                 "LOCO fitted prediction population/order differs")
        receipt = {**receipt, "slot_id": slot["slot_id"], "domain": "loco", "candidate_id": cid,
                   "alpha": alpha, "pipeline": slot["pipeline"], "representation": slot["representation"],
                   "heldout_component": heldout, "seed": 20261017, "new_outer_fits": 1,
                   "full_training_records": int(train.sum()), "supported_training_records": int(support.sum()),
                   "test_pairs": len(raw), "test_pair_ids": test["pair_ids"].tolist(),
                   "training_support_rule": "full train then retain blocks with at least two records",
                   "feature_binding": feature_info, "model_adapter": str(Path(__file__).resolve().parents[1] / "core/model_adapters.py")}
        joblib.dump(model, model_path)
        temporary = receipt_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        temporary.replace(receipt_path)
        return {"raw": raw, "fit_receipt": receipt, "model_path": str(model_path)}
