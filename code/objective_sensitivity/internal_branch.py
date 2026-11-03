"""Read-only Internal identity-split caches and one-fit Ridge adaptation for objective sensitivity.

No old queue, runner, checkpoint recovery, or model loading is invoked.  The
controller owns objective-specific calibration/selection, source hashes,
resource admission, operation accounting, and final acceptance.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path


ALPHAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
CANDIDATES = tuple(f"ridge_{i:02d}" for i in range(8))


def require(ok, message):
    if not ok:
        raise RuntimeError("INTERNAL_SENSITIVITY: " + message)


class InternalBranch:
    def __init__(self, candidate_root=None, queue_root=None, component_map=None, output_root=None,
                 manifest=None):
        import numpy as np
        import pandas as pd

        self.np, self.pd = np, pd
        normalized = None
        if manifest is not None:
            manifest_path = Path(manifest).resolve()
            normalized = json.loads(manifest_path.read_text(encoding="utf-8"))
            def resolve(value):
                path = Path(value)
                return (path if path.is_absolute() else manifest_path.parent / path).resolve()
            candidate_root = resolve(normalized["data_dir"])
            component_map = candidate_root / "pair_dependency_components.csv"
            queue_root = manifest_path.parent
            for job in normalized["jobs"]:
                job["cell_path"] = str(resolve(job["cell_path"]))
        self.candidate_root = Path(candidate_root).resolve()
        self.queue_root = Path(queue_root).resolve()
        self.output_root = Path(output_root).resolve()
        self.source_paths = set()
        self.spec = {"jobs": normalized["jobs"]} if normalized is not None else self._json(self.candidate_root / "spec.json")
        self.cfg = self._json(Path(__file__).resolve().parents[1] / "core/scientific_config.json") if normalized is not None else self._json(self.candidate_root / "scientific_config.json")
        require(tuple(self.cfg["learners"]["ridge"]["alphas"]) == ALPHAS, "Ridge alpha grid drift")
        self.paths = ({key:self.candidate_root / value for key,value in {
            "records":"records_revised.csv", "pairs":"pairs_revised.csv", "fingerprints":"fingerprints.csv",
            "count_cache":"main_raw_sparse_counts.json", "outer_folds":"outer_identity_folds.csv",
            "inner_folds":"inner_identity_folds.csv", "vocab_dir":"vocab"}.items()}
            if normalized is not None else {key:self.candidate_root / value for key,value in self.spec["inputs"].items()})
        self.source_paths.update(str(path.resolve()) for path in self.paths.values() if path.is_file())
        self.source_paths.add(str(Path(__file__).resolve().parents[1] / "core/model_adapters.py"))
        self.component_path = Path(component_map).resolve()
        self.records = self._csv(self.paths["records"])
        if "primary_pair_eligible" in self.records:
            tokens = self.records.primary_pair_eligible.astype(str).str.strip().str.lower()
            require(tokens.isin(["true", "false", "1", "0"]).all(), "invalid eligibility token")
            self.records = self.records.loc[tokens.isin(["true", "1"])].copy()
        self.records = self.records.sort_values("record_id", kind="mergesort").reset_index(drop=True)
        self.pairs = self._csv(self.paths["pairs"]).sort_values("pair_id", kind="mergesort").reset_index(drop=True)
        require(len(self.records) == 699 and self.records.identity_hash.nunique() == 511
                and len(self.pairs) == 5804, "corrected population is not 699/511/5804")
        require(not self.records.record_id.duplicated().any() and not self.pairs.pair_id.duplicated().any(), "duplicate record/pair IDs")
        index = dict(zip(self.records.record_id, range(len(self.records))))
        require(set(self.pairs.record_id_i) | set(self.pairs.record_id_j) <= set(index), "pair endpoint missing")
        self.pi = np.asarray([index[rid] for rid in self.pairs.record_id_i], dtype=np.int64)
        self.pj = np.asarray([index[rid] for rid in self.pairs.record_id_j], dtype=np.int64)
        for indices, column in ((self.pi, "identity_hash_i"), (self.pj, "identity_hash_j")):
            require(np.array_equal(self.records.iloc[indices].identity_hash.to_numpy(), self.pairs[column].to_numpy()), "pair identity direction drift")
        for column in ("normalized_poi_id", "pair_context_hash"):
            require(not self.records[column].isna().any(), "missing endpoint block key")
            require(np.array_equal(self.records.iloc[self.pi][column].to_numpy(), self.records.iloc[self.pj][column].to_numpy()), "pair crosses endpoint blocks")
        self.z = pd.to_numeric(self.records.pdc50_median).to_numpy(dtype=np.float64)
        self.y = self.z[self.pj] - self.z[self.pi]
        require(np.isfinite(self.z).all() and np.isfinite(self.y).all(), "nonfinite labels")
        require(np.max(np.abs(self.y + pd.to_numeric(self.pairs.true_delta).to_numpy(dtype=np.float64))) <= 1e-12,
                "canonical label must be b-minus-a / negative source true_delta")
        components = self._csv(self.component_path)
        require(not components.pair_id.duplicated().any() and set(components.pair_id) == set(self.pairs.pair_id), "component pair universe drift")
        components = components.set_index("pair_id").loc[self.pairs.pair_id]
        for column in ("record_id_i", "record_id_j"):
            require(np.array_equal(components[column].to_numpy(), self.pairs[column].to_numpy()), "component endpoint orientation drift")
        require(not components.dependency_component.isna().any(), "missing component ID")
        self.component_ids = components.dependency_component.to_numpy(dtype=str)
        self.outer = self._csv(self.paths["outer_folds"])
        self.inner = self._csv(self.paths["inner_folds"])
        self._slots, self._by_id, self._fold_cache, self._test_cache = [], {}, {}, {}
        self._binary_X = self._raw_counts = self._adapters = None
        for number, job in enumerate(self.spec["jobs"]):
            if job["learner"] != "ridge" or job["representation"] not in ("binary", "count") or job["pipeline"] not in ("endpoint_E1", "endpoint_E0_matched"):
                continue
            slot_id = f"internal_{number:04d}"
            cell = Path(job["cell_path"]) if "cell_path" in job else self.queue_root / "parent_slots" / slot_id / job["representation"] / "results" / f"ridge_R{job['repeat']}" / f"outer_{job['outer_fold']}" / job["pipeline"]
            selection = self._json(cell / "selection.json")
            require(selection["selection_metric"] == "calibrated inner-OOF pair MAE" and selection["tie_tolerance"] == 1e-12, "old selection convention drift")
            scores = {}
            for row in selection["candidates"]:
                cid = row["candidate_id"]
                require(cid in CANDIDATES and cid not in scores and row["params"] == {"alpha": ALPHAS[CANDIDATES.index(cid)]}, "candidate parameter identity drift")
                scores[cid] = {"scale_c": float(row["scale_c"]), "selection_score": float(row["inner_oof_calibrated_mae"])}
                self.source_paths.update(str(cell / "candidates" / cid / name) for name in ("candidate_complete.json", "oof_predictions.npz"))
            require(set(scores) == set(CANDIDATES), "missing/extra old candidate scores")
            selected = selection["selected"]
            require(selected["candidate_id"] in scores, "unknown selected candidate")
            slot = {"slot_id": slot_id, "domain": "internal", "representation": job["representation"],
                    "pipeline": job["pipeline"], "repeat": int(job["repeat"]), "outer_fold": int(job["outer_fold"]),
                    "old_candidate_id": selected["candidate_id"], "old_scale": float(selected["scale_c"]),
                    "old_scores": scores, "model_seed": int(self.cfg["splitting"]["model_seeds_paired_by_repeat"][job["repeat"]]),
                    "cell_path": str(cell)}
            require(slot["old_scale"] == scores[slot["old_candidate_id"]]["scale_c"], "selected scale mismatch")
            require(all(np.isfinite(value) for item in scores.values() for value in item.values()), "nonfinite old scores")
            self._slots.append(slot)
            self._by_id[slot_id] = slot
            self.source_paths.add(str(cell / "outer_predictions.csv"))
            if job["representation"] == "count":
                self.source_paths.add(str(self.paths["vocab_dir"] / f"R{job['repeat']}_O{job['outer_fold']}_I-1.npz"))
        require(len(self._slots) == 60 and len(self._by_id) == 60, "Internal slot census is not 60")

    def _json(self, path):
        path = Path(path).resolve()
        self.source_paths.add(str(path))
        return json.loads(path.read_text(encoding="utf-8"))

    def _csv(self, path):
        path = Path(path).resolve()
        self.source_paths.add(str(path))
        return self.pd.read_csv(path)

    def slots(self):
        return [dict(slot, old_scores={key: dict(value) for key, value in slot["old_scores"].items()}) for slot in self._slots]

    def _slot(self, slot):
        original = self._by_id[slot["slot_id"]]
        require(all(slot[key] == original[key] for key in ("domain", "representation", "pipeline", "repeat", "outer_fold", "old_candidate_id", "old_scale")), "slot identity changed")
        return original

    def _folds(self, slot):
        np = self.np
        key = (slot["repeat"], slot["outer_fold"])
        if key not in self._fold_cache:
            outer = self.outer.loc[self.outer.repeat.eq(key[0])]
            inner = self.inner.loc[self.inner.repeat.eq(key[0]) & self.inner.outer_fold.eq(key[1])]
            require(not outer.identity_hash.duplicated().any() and not inner.identity_hash.duplicated().any(), "duplicate identity-fold assignment")
            outer_map = dict(zip(outer.identity_hash, outer.outer_fold))
            inner_map = dict(zip(inner.identity_hash, inner.inner_fold))
            require(set(self.records.identity_hash) <= set(outer_map), "missing outer identity assignment")
            require(set(inner_map.values()) == {0, 1, 2}, "inner fold universe drift")
            record_outer = self.records.identity_hash.map(outer_map).to_numpy()
            record_train = record_outer != key[1]
            require(not any(outer_map.get(identity) == key[1] for identity in inner_map), "outer test identity in inner folds")
            require(set(self.records.loc[record_train, "identity_hash"]) <= set(inner_map), "missing training inner assignment")
            pii = self.pairs.identity_hash_i.map(inner_map).fillna(-1).to_numpy(dtype=np.int64)
            pjj = self.pairs.identity_hash_j.map(inner_map).fillna(-1).to_numpy(dtype=np.int64)
            oof = np.flatnonzero((pii >= 0) & (pii == pjj))
            test = np.flatnonzero((record_outer[self.pi] == key[1]) & (record_outer[self.pj] == key[1]))
            require(len(oof) > 0 and len(test) > 0 and not np.intersect1d(oof, test).size, "empty/leaking OOF/test population")
            self._fold_cache[key] = (record_train, oof, test)
        return self._fold_cache[key]

    def _result(self, indices, raw):
        np = self.np
        raw = np.asarray(raw, dtype=np.float64)
        require(raw.ndim == 1 and len(raw) == len(indices) and np.isfinite(raw).all(), "invalid raw prediction")
        return {"pair_ids": self.pairs.iloc[indices].pair_id.to_numpy(dtype=str),
                "component_ids": self.component_ids[indices].copy(), "y": self.y[indices].copy(), "raw": raw.copy(),
                "record_id_i": self.pairs.iloc[indices].record_id_i.to_numpy(dtype=str),
                "record_id_j": self.pairs.iloc[indices].record_id_j.to_numpy(dtype=str)}

    def load_oof(self, slot, candidate_id):
        slot = self._slot(slot)
        require(candidate_id in CANDIDATES, "unknown OOF candidate")
        folder = Path(slot["cell_path"]) / "candidates" / candidate_id
        meta = self._json(folder / "candidate_complete.json")
        require(meta["candidate_id"] == candidate_id and meta["params"] == {"alpha": ALPHAS[CANDIDATES.index(candidate_id)]}, "candidate OOF metadata drift")
        expected = slot["old_scores"][candidate_id]
        require(meta["scale_c"] == expected["scale_c"] and meta["inner_oof_calibrated_mae"] == expected["selection_score"], "candidate/selection metadata disagreement")
        path = folder / "oof_predictions.npz"
        with self.np.load(path, allow_pickle=False) as cache:
            require(set(cache.files) == {"pair_indices", "raw_prediction"}, "unexpected OOF NPZ fields")
            indices, raw = cache["pair_indices"].copy(), cache["raw_prediction"].copy()
        require(indices.ndim == 1 and indices.dtype.kind in "iu" and self.np.array_equal(indices, self._folds(slot)[1]), "OOF indices differ from original sorted inner population")
        require(int(meta["inner_oof_pairs"]) == len(indices), "OOF metadata census mismatch")
        return self._result(indices, raw)

    def load_test(self, slot):
        slot = self._slot(slot)
        if slot["slot_id"] not in self._test_cache:
            table = self._csv(Path(slot["cell_path"]) / "outer_predictions.csv").sort_values("pair_id", kind="mergesort").reset_index(drop=True)
            indices = self._folds(slot)[2]
            expected = self.pairs.iloc[indices]
            require(not table.pair_id.duplicated().any() and self.np.array_equal(table.pair_id.to_numpy(), expected.pair_id.to_numpy()), "outer prediction pair population drift")
            for column in ("record_id_i", "record_id_j"):
                require(self.np.array_equal(table[column].to_numpy(), expected[column].to_numpy()), "outer prediction direction drift")
            require(self.np.max(self.np.abs(table.true_delta.to_numpy(dtype=float) - self.y[indices])) <= 1e-12, "outer prediction label mismatch")
            require(table.selected_candidate_id.eq(slot["old_candidate_id"]).all() and (self.np.abs(table.scale_c.to_numpy(dtype=float) - slot["old_scale"]) <= 1e-12).all(), "outer prediction selection/scale mismatch")
            require(self.np.array_equal(table.pair_row_index.to_numpy(), indices), "outer prediction global index mismatch")
            self._test_cache[slot["slot_id"]] = self._result(indices, table.raw_prediction.to_numpy(dtype=float))
        return {key: value.copy() for key, value in self._test_cache[slot["slot_id"]].items()}

    def _features(self, slot, record_train):
        np = self.np
        if slot["representation"] == "binary":
            if self._binary_X is None:
                fps = self._csv(self.paths["fingerprints"])
                require(not fps.record_id.duplicated().any(), "duplicate fingerprint record")
                mapping = fps.set_index("record_id").fp_hex
                rows = []
                for record_id in self.records.record_id:
                    token = mapping.at[record_id]
                    require(isinstance(token, str) and len(token) == 512 and token == token.upper(), "invalid binary fingerprint hex")
                    rows.append(np.unpackbits(np.frombuffer(bytes.fromhex(token)[::-1], dtype=np.uint8), bitorder="little").astype(np.float32, copy=False))
                self._binary_X = np.vstack(rows)
            return self._binary_X, {"representation": "folded_binary", "feature_columns": 2048}
        if self._raw_counts is None:
            cache = self._json(self.paths["count_cache"])
            rows = cache["records"]
            require(len({row["record_id"] for row in rows}) == len(rows), "duplicate raw count record")
            self._raw_counts = {row["record_id"]: {int(key): int(value) for key, value in row["counts"]} for row in rows}
        layer = f"R{slot['repeat']}_O{slot['outer_fold']}_I-1"
        path = self.paths["vocab_dir"] / (layer + ".npz")
        with np.load(path, allow_pickle=False) as values:
            vocab, training_ids = values["environment_ids"].copy(), values["training_record_ids"].copy()
        require(np.array_equal(training_ids, self.records.loc[record_train, "record_id"].to_numpy(str)), "outer count vocabulary training scope mismatch")
        require(vocab.ndim == 1 and len(np.unique(vocab)) == len(vocab), "invalid outer count vocabulary")
        require(vocab.tolist() == sorted({environment for record_id in training_ids
                                          for environment in self._raw_counts[record_id]}),
                "outer count vocabulary is not the training-only environment union")
        # Same float32 mapping, original vocabulary order, and drop-unknown rule
        # as frozen count_features.map_counts; no new environment is inserted.
        columns = {int(key): column for column, key in enumerate(vocab)}
        X = np.zeros((len(self.records), len(vocab)), dtype=np.float32)
        for row, record_id in enumerate(self.records.record_id):
            for key, count in self._raw_counts[record_id].items():
                column = columns.get(key)
                if column is not None:
                    X[row, column] = count
        return X, {"representation": "unfolded_raw_sparse_count", "vocabulary_layer": layer,
                   "vocabulary_path": str(path), "vocabulary_size": len(vocab), "vocabulary_training_records": len(training_ids),
                   "normalization": "none", "unknown_environment_rule": "drop; no heldout vocabulary extension"}

    def fit_predict(self, slot, candidate_id, output_dir):
        """Perform exactly one new outer Ridge fit when called by the controller."""
        import joblib
        from threadpoolctl import threadpool_limits

        slot = self._slot(slot)
        require(candidate_id in CANDIDATES and candidate_id != slot["old_candidate_id"], "refit requires a different selected candidate")
        directory = Path(output_dir).resolve()
        require(directory.is_relative_to(self.output_root) and directory != self.output_root, "output escapes objective sensitivity")
        directory.mkdir(parents=True, exist_ok=True)
        model_path, receipt_path = directory / "model.joblib", directory / "receipt.json"
        require(not model_path.exists() and not receipt_path.exists(), "existing fit output; no automatic refit/resume")
        record_train, _, test_indices = self._folds(slot)
        self.load_test(slot)  # Validate exact test ordering before a model is fit.
        X, feature_info = self._features(slot, record_train)
        positions = self.np.flatnonzero(record_train)
        frame = self.records.iloc[positions]
        blocks = (frame.normalized_poi_id.astype(str) + "\x1f" + frame.pair_context_hash.astype(str)).to_numpy(dtype=object)
        counts = self.pd.Series(blocks).value_counts()
        keep = self.np.asarray([counts[value] >= 2 for value in blocks], dtype=bool)
        require(keep.any(), "no nonsingleton outer endpoint support")
        positions, blocks = positions[keep], blocks[keep]
        stats = {"records_before_singleton_filter": int(record_train.sum()), "records_after_singleton_filter": len(positions),
                 "singleton_blocks_excluded": int((counts == 1).sum()), "retained_blocks": int((counts >= 2).sum())}
        if self._adapters is None:
            path = Path(__file__).resolve().parents[1] / "core/model_adapters.py"
            spec = importlib.util.spec_from_file_location("_objective_internal_frozen_model_adapters", path)
            require(spec is not None and spec.loader is not None, "cannot import frozen model adapters")
            self._adapters = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self._adapters)
        alpha = ALPHAS[CANDIDATES.index(candidate_id)]
        with threadpool_limits(limits=4):
            if slot["pipeline"] == "endpoint_E1":
                model, scientific_receipt = self._adapters.fit_ridge_e1_within(alpha, X[positions], self.z[positions], blocks)
            else:
                model, scientific_receipt = self._adapters.fit_model("ridge", {"alpha": alpha, "fit_intercept": True},
                                                                         X[positions], self.z[positions], slot["model_seed"], "cpu")
            raw = self._adapters.predict_model(model, X[self.pj[test_indices]]) - self._adapters.predict_model(model, X[self.pi[test_indices]])
        require(raw.shape == (len(test_indices),) and self.np.isfinite(raw).all(), "invalid new outer raw prediction")
        receipt = {"slot_id": slot["slot_id"], "domain": "internal", "representation": slot["representation"],
                   "pipeline": slot["pipeline"], "repeat": slot["repeat"], "outer_fold": slot["outer_fold"],
                   "candidate_id": candidate_id, "alpha": alpha, "model_seed": slot["model_seed"], "fit_count": 1,
                   "fit_scope": "original full outer training, same nonsingleton endpoint support", "threads": 4,
                   "support": stats, "features": feature_info, "scientific_receipt": scientific_receipt,
                   "test_pairs": len(test_indices), "test_order": "pair_id stable ascending; identical to load_test", "model_path": str(model_path)}
        temporary = model_path.with_name(model_path.name + f".tmp.{os.getpid()}")
        joblib.dump(model, temporary, compress=1)
        os.replace(temporary, model_path)
        temporary = receipt_path.with_name(receipt_path.name + f".tmp.{os.getpid()}")
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, receipt_path)
        return {"raw": raw, "fit_receipt": receipt, "model_path": str(model_path)}
