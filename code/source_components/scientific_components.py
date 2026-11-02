"""Source-component data and prediction routes from the frozen implementation.

HELDOUT_COMPONENT is bound by the caller before bundle construction. Input
tables and vocabularies are supplied locally; no training happens on import.
"""
from __future__ import annotations
from collections import Counter
from types import SimpleNamespace
from pathlib import Path
from typing import Any
import hashlib
import itertools
import json
import numpy as np
import pandas as pd
from model_adapters import predict_model
HELDOUT_COMPONENT = None

def require(value, message):
    if not value:
        raise RuntimeError(message)

def sha(value):
    return hashlib.sha256(value).hexdigest().upper()

def js(path):
    return json.loads(Path(path).read_bytes())

def json_signature(value):
    return sha(json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode())

class PairFeatures:
    def __init__(self, X, ii, jj): self.X, self.ii, self.jj = X, ii, jj
    def __getitem__(self, indices): return self.X[self.jj[indices]] - self.X[self.ii[indices]]

def bundle(base, new, mapper):
    d = base if new else base / 'data'; c = base if new else base / 'config'
    r = pd.read_csv(d / 'records_revised.csv').sort_values('record_id', kind='mergesort').reset_index(drop=True)
    p = pd.read_csv(d / 'pairs_revised.csv').sort_values('pair_id', kind='mergesort').reset_index(drop=True)
    g = pd.read_csv(d / 'record_source_group_map.csv').set_index('record_id').component_id.to_dict()
    im = pd.read_csv(c / 'inner_component_folds.csv'); vm = js(c / 'vocab_manifest.json')
    folds = im[im.outer_group_id.eq(HELDOUT_COMPONENT)].set_index('component_id').inner_fold.astype(int).to_dict()
    rids = r.record_id.tolist(); ix = {rid: i for i, rid in enumerate(rids)}
    require(len(r) == len(ix) == 699 and r.primary_pair_eligible.all(), 'Record population/eligibility changed')
    ii = np.array([ix[x] for x in p.record_id_i], np.int64); jj = np.array([ix[x] for x in p.record_id_j], np.int64)
    rg = np.array([g[x] for x in rids]); pg = rg[ii]
    require(np.array_equal(pg, rg[jj]) and len(np.unique(pg)) == 43, 'Pair components differ')
    rf = np.array([folds.get(x, -1) for x in rg], np.int64)
    require(np.all((rf < 0) == (rg == HELDOUT_COMPONENT)) and set(rf) == {-1, 0, 1, 2}, 'Inner assignment incomplete')
    z = r.pdc50_median.to_numpy(np.float64); y = z[jj] - z[ii]
    authority = p['canonical_target_b_minus_a' if new else 'true_delta'].to_numpy(np.float64)
    require(np.max(np.abs(y - (authority if new else -authority))) <= 1e-12, 'Target direction differs')
    require(r.identity_hash.to_numpy()[ii].tolist() == p.identity_hash_i.tolist()
            and r.identity_hash.to_numpy()[jj].tolist() == p.identity_hash_j.tolist(), 'Pair identity differs')
    for col in ['normalized_poi_id', 'pair_context_hash']:
        require(np.array_equal(r[col].to_numpy()[ii], r[col].to_numpy()[jj]), 'Incompatible pair context')
    fp = pd.read_csv(d / 'fingerprints.csv').set_index('record_id').fp_hex.to_dict()
    xb = np.stack([np.unpackbits(np.frombuffer(bytes.fromhex(fp[x])[::-1], dtype=np.uint8), bitorder='little').astype(np.float32) for x in rids])
    require(xb.shape == (699, 2048), 'Binary feature width differs')
    counts = {x['record_id']: {int(k): int(v) for k, v in x['counts']} for x in js(d / 'main_raw_sparse_counts.json')['records']}
    blocks = (r.normalized_poi_id.astype(str) + '\x1f' + r.pair_context_hash.astype(str)).to_numpy(object)
    xs, layer_info = {}, []
    for layer in ([-1, 0, 1, 2] if new else [0]):
        key = f'R0_G{HELDOUT_COMPONENT}_I{layer}'; item = vm[key]; vp = (c / item['path']).resolve()
        require(sha(vp.read_bytes()) == item['sha256'], 'Manifest vocabulary SHA mismatch')
        with np.load(vp, allow_pickle=False) as arr: env, tids = arr['environment_ids'].copy(), arr['training_record_ids'].copy()
        train = (rf >= 0) if layer == -1 else ((rf >= 0) & (rf != layer))
        require(np.array_equal(tids, r.record_id.to_numpy()[train]), 'Vocabulary training row order differs')
        require(env.tolist() == sorted({e for rid in tids for e in counts[rid]}), 'Vocabulary scope differs')
        xc, _ = mapper([counts[x] for x in rids], env)
        xs[layer] = {'binary': xb, 'count': xc}
        layer_info.append({'inner_fold': layer, 'training_records': int(train.sum()), 'count_columns': len(env),
                           'vocabulary': vp.as_posix(), 'ordered_training_ids_sha256': json_signature(tids.tolist())})
    return SimpleNamespace(records=r, pairs=p, ii=ii, jj=jj, z=z, y=y, rg=rg, pg=pg, rf=rf, blocks=blocks, xs=xs, layers=layer_info)

def masks(b, layer):
    train = b.rf >= 0 if layer == -1 else (b.rf >= 0) & (b.rf != layer)
    pairs = train[b.ii] & train[b.jj]
    valid = b.rg == HELDOUT_COMPONENT if layer == -1 else b.rf == layer
    idx = np.flatnonzero(valid[b.ii] & valid[b.jj])
    require(pairs.any() and len(idx), 'Empty train/validation')
    bc = Counter(b.blocks[train]); support = train & np.array([bc.get(x, 0) >= 2 for x in b.blocks])
    return train, pairs, support, idx

def training(b, rep, pipeline, layer):
    train, pairs, support, _ = masks(b, layer); X = b.xs[layer][rep]
    if pipeline == 'direct_pair':
        d = X[b.jj[pairs]] - X[b.ii[pairs]]
        return np.vstack([d, -d]), np.concatenate([b.y[pairs], -b.y[pairs]]), None
    positions = train if pipeline == 'endpoint_E0' else support
    require(positions.any(), 'Empty endpoint support')
    return X[positions], b.z[positions], b.blocks[positions] if pipeline == 'endpoint_E1' else None

def predict(b, rep, pipeline, layer, model, idx, route):
    X = b.xs[layer][rep]
    pure = SimpleNamespace(X_record=X, X_diff=PairFeatures(X, b.ii, b.jj), pair_i=b.ii, pair_j=b.jj)
    return route(model, pipeline, pure, idx)

def map_counts(counts: list[dict[int, int]], vocab: np.ndarray):
    column = {int(k): p for p, k in enumerate(vocab)}
    X = np.zeros((len(counts), len(vocab)), np.float32)
    unknown = []
    for i, count in enumerate(counts):
        lost_ids, lost_counts = 0, 0
        for k, value in count.items():
            pos = column.get(k)
            if pos is None:
                lost_ids += 1
                lost_counts += value
            else:
                X[i, pos] = value
        unknown.append({"unknown_environment_ids": lost_ids, "raw_environment_ids": len(count),
                        "unknown_environment_count": lost_counts, "raw_environment_count": sum(count.values()),
                        "unknown_id_fraction": lost_ids / len(count) if count else 0.0,
                        "unknown_count_fraction": lost_counts / sum(count.values()) if count else 0.0})
    return X, pd.DataFrame(unknown)

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
