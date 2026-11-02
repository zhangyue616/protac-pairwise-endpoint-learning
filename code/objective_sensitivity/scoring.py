"""Frozen per-candidate objective scoring and deterministic selection."""
from __future__ import annotations
import math
import numpy as np
TOL = 1e-12

def require(value, message):
    if not value:
        raise RuntimeError(message)

def validate_population(data, name):
    for field in ('pair_ids', 'component_ids', 'y', 'raw'):
        require(field in data, name+': missing '+field)
    ids = np.asarray(data['pair_ids']).astype(str)
    groups = np.asarray(data['component_ids']).astype(str)
    y, raw = np.asarray(data['y'], np.float64), np.asarray(data['raw'], np.float64)
    require(ids.ndim == 1 and len(ids) > 0 and len(set(ids)) == len(ids), name+': nonunique/empty pairs')
    require(groups.shape == y.shape == raw.shape == ids.shape, name+': shape mismatch')
    require(np.isfinite(y).all() and np.isfinite(raw).all(), name+': nonfinite')
    require(all(x and x != 'nan' for x in groups), name+': missing component')
    return ids, groups, y, raw

def evaluate(h, y, raw, groups, objective):
    labels = groups if objective == 'component_macro' else np.full(len(y), 'ALL', dtype=object)
    n_groups = len(np.unique(labels))
    c = h.nonnegative_component_l1_scale(y, raw, labels, expected_components=n_groups)
    error = np.abs(y-c*raw)
    require(np.isfinite(error).all(), 'Nonfinite calibrated error')
    _, codes, counts = np.unique(labels, return_inverse=True, return_counts=True)
    score = math.fsum(float(v) for v in error/(n_groups*counts[codes]))
    return {'scale_c': float(c), 'selection_score': float(score)}

def winner(rows):
    best = min(x['selection_score'] for x in rows)
    return min((x for x in rows if x['selection_score'] <= best+TOL), key=lambda x:x['candidate_id'])
