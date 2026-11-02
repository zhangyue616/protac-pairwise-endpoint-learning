"""Retained seven-compound ensemble and point-estimate formulas."""
import numpy as np
import pandas as pd
TOL = 1e-12

def require(ok, message):
    if not bool(ok):
        raise RuntimeError(message)

def metrics(y, p):
    y, p = (np.asarray(y, float), np.asarray(p, float))
    require(np.isfinite(y).all() and np.isfinite(p).all(), 'nonfinite metric input')
    require(not ((np.abs(y) <= TOL) & (y != 0)).any(), 'near-zero nonzero truth requires explicit direction policy resolution')
    nz = y != 0
    pzero = np.abs(p[nz]) <= TOL
    correct = (np.sign(y[nz]) == np.sign(p[nz])) & ~pzero
    return {'n': len(y), 'n_direction': int(nz.sum()), 'mae': float(np.mean(np.abs(y - p))) if len(y) else None, 'rmse': float(np.sqrt(np.mean((y - p) ** 2))) if len(y) else None, 'DA_strict': float(correct.mean()) if nz.any() else None, 'DA_half': float(np.mean(correct.astype(float) + 0.5 * pzero)) if nz.any() else None, 'zero_predictions': int((np.abs(p) <= TOL).sum()), 'true_ties_excluded': int((~nz).sum())}

def summarize_external(members):
    require(not members.duplicated(['representation', 'learner', 'pipeline', 'repeat', 'outer_fold', 'pair_id']).any(), 'external member key duplicated')
    require(np.allclose(members.raw_prediction * members.scale_c, members.calibrated_prediction, rtol=TOL, atol=TOL), 'external scale mismatch')
    keys = ['representation', 'learner', 'pipeline', 'pair_id']
    rows = []
    for key, g in members.groupby(keys, sort=True):
        require(len(g) == 15 and set(zip(g.repeat, g.outer_fold)) == {(r, f) for r in range(3) for f in range(5)}, 'external member set incomplete')
        require(g.true_delta.nunique() == 1, 'external label drift')
        rows.append({**dict(zip(keys, key)), 'record_id_i': g.record_id_i.iloc[0], 'record_id_j': g.record_id_j.iloc[0], 'true_delta': g.true_delta.iloc[0], 'raw_prediction': g.raw_prediction.mean(), 'calibrated_prediction': g.calibrated_prediction.mean(), 'members': 15, 'evidence_level': 'previously inspected exploratory single series'})
    ensemble = pd.DataFrame(rows)
    metric = []
    for key, g in ensemble.groupby(['representation', 'learner', 'pipeline']):
        for scale, col in [('raw', 'raw_prediction'), ('calibrated', 'calibrated_prediction')]:
            metric.append({**dict(zip(['representation', 'learner', 'pipeline'], key)), 'scale': scale, **metrics(g.true_delta, g[col])})
        if key[2] == 'direct_pair':
            metric.append({'representation': key[0], 'learner': key[1], 'pipeline': 'zero', 'scale': 'raw', **metrics(g.true_delta, np.zeros(len(g)))})
    return (ensemble, pd.DataFrame(metric))
