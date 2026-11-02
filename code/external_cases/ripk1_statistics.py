"""Retained ten-compound ensemble, layer metrics and contrast formulas."""
import numpy as np
import pandas as pd
TOL = 1e-12
REPS = ["binary", "count"]
LEARNERS = ["ridge", "random_forest", "xgboost"]
ARMS = ["direct_pair", "endpoint_E0", "endpoint_E1", "endpoint_E0_matched", "context_pair_C1"]

class GateError(RuntimeError):
    pass

def require(ok, reason):
    if not bool(ok):
        raise GateError(reason)

def combos():
    return [(rep, learner, arm) for rep in REPS for learner in LEARNERS for arm in ARMS if not (learner == 'ridge' and arm == 'context_pair_C1')]

def metrics(y, prediction, np):
    require(np.isfinite(y).all() and np.isfinite(prediction).all(), 'nonfinite metric input')
    require(not ((np.abs(y) <= TOL) & (y != 0)).any(), 'near-zero nonzero truth: explicit direction policy required')
    nonzero = y != 0
    pzero = np.abs(prediction[nonzero]) <= TOL
    correct = (np.sign(y[nonzero]) == np.sign(prediction[nonzero])) & ~pzero
    return {'n_pairs': len(y), 'n_direction': int(nonzero.sum()), 'MAE': float(np.mean(np.abs(y - prediction))), 'RMSE': float(np.sqrt(np.mean((y - prediction) ** 2))), 'strict_DA': float(correct.mean()) if nonzero.any() else None, 'half_zero_DA': float(np.mean(correct.astype(float) + 0.5 * pzero)) if nonzero.any() else None, 'zero_predictions': int((np.abs(prediction) <= TOL).sum()), 'true_ties_excluded': int((~nonzero).sum())}

def validate_model(model, member):
    width = member['feature_columns']
    require(hasattr(model, 'n_features_in_') and int(model.n_features_in_) == width, 'actual loaded model feature dimension mismatch')
    if hasattr(model, 'get_booster'):
        require(model.get_booster().num_features() == width, 'XGBoost booster feature dimension mismatch')
    if hasattr(model, 'coef_'):
        require(model.coef_.shape[-1] == width, 'Ridge coefficient feature dimension mismatch')
    params = model.get_params()
    for key, value in member['selected_params'].items():
        require(params.get(key) == value, 'loaded model differs from fixed selected candidate: ' + key)
    return {'member_id': member['member_id'], 'model_sha256': member['model_sha256'], 'representation': member['representation'], 'learner': member['learner'], 'strategy': member['strategy'], 'repeat': member['repeat'], 'outer_fold': member['outer_fold'], 'expected_feature_columns': width, 'actual_feature_columns': int(model.n_features_in_), 'dimension_status': 'PASS', 'feature_column_order': 'fixed2048 bit indices' if member['representation'] == 'binary' else 'frozen outer environment_ids order', 'pair_feature_order': 'difference then sum' if member['strategy'] == 'context_pair_C1' else 'difference or endpoint', 'new_poi_encoding': 'none; structure-only features', 'fallback_used': False, 'selected_candidate_status': 'PASS'}

def summarize_members(members_frame, pair_meta, primary, bases):
    ensemble = []
    keys = ['representation', 'learner', 'strategy', 'pair_id']
    for key, group in members_frame.groupby(keys, sort=True):
        require(len(group) == 15 and group.member_id.nunique() == 15 and (group.true_delta.nunique() == 1), 'ensemble must retain all15 fixed members')
        first = group.iloc[0]
        ensemble.append(dict(zip(keys, key)) | {'record_id_i': first.record_id_i, 'record_id_j': first.record_id_j, 'primary_ge05': bool(first.primary_ge05), 'true_delta': float(first.true_delta), 'raw_prediction': float(group.raw_prediction.mean()), 'calibrated_prediction': float(group.calibrated_prediction.mean()), 'model_members': 15})
    for index, pair in pair_meta.iterrows():
        ensemble.append({'representation': 'reference', 'learner': 'none', 'strategy': 'Zero', 'pair_id': pair.pair_id, 'record_id_i': pair.record_id_i, 'record_id_j': pair.record_id_j, 'primary_ge05': bool(primary[index]), 'true_delta': float(bases['binary'].y_pair[index]), 'raw_prediction': 0.0, 'calibrated_prediction': 0.0, 'model_members': 0})
    ensemble_frame = pd.DataFrame(ensemble)
    metric_rows = []
    for key, group in ensemble_frame.groupby(['representation', 'learner', 'strategy'], sort=True):
        for layer in ['primary42', 'secondary45']:
            selected = group.loc[group.primary_ge05] if layer == 'primary42' else group
            require(len(selected) == (42 if layer == 'primary42' else 45), 'metric layer support drift')
            for scale, column in [('raw', 'raw_prediction'), ('calibrated', 'calibrated_prediction')]:
                metric_rows.append(dict(zip(['representation', 'learner', 'strategy'], key)) | {'layer': layer, 'output_scale': scale, 'source_family_id': 'RIPK1_WO2022120118_FAMILY', 'source_families': 1, 'condition_blocks': 1} | metrics(selected.true_delta.to_numpy(float), selected[column].to_numpy(float), np))
    require(len(metric_rows) == 116, 'complete28+Zero metrics census mismatch')
    metric_map = {(r['representation'], r['learner'], r['strategy'], r['layer'], r['output_scale']): r for r in metric_rows}
    contrasts = []
    for layer in ['primary42', 'secondary45']:
        for scale in ['raw', 'calibrated']:
            for rep in REPS:
                for learner in LEARNERS:
                    for label, arm in [('E0matched-E1', 'endpoint_E0_matched'), ('Direct-E1', 'direct_pair')]:
                        lhs = metric_map[rep, learner, arm, layer, scale]
                        rhs = metric_map[rep, learner, 'endpoint_E1', layer, scale]
                        contrasts.append({'contrast': label, 'representation': rep, 'learner': learner, 'strategy': arm, 'layer': layer, 'output_scale': scale, 'lhs_strategy': arm, 'rhs_strategy': 'endpoint_E1', 'MAE_difference': lhs['MAE'] - rhs['MAE'], 'positive_favors': 'endpoint_E1'})
            zero = metric_map['reference', 'none', 'Zero', layer, scale]
            for rep, learner, arm in combos():
                lhs = metric_map[rep, learner, arm, layer, scale]
                contrasts.append({'contrast': 'Model-Zero', 'representation': rep, 'learner': learner, 'strategy': arm, 'layer': layer, 'output_scale': scale, 'lhs_strategy': arm, 'rhs_strategy': 'Zero', 'MAE_difference': lhs['MAE'] - zero['MAE'], 'positive_favors': 'Zero'})
    require(len(contrasts) == 160, 'contrast census mismatch')
    return (ensemble_frame, pd.DataFrame(metric_rows), pd.DataFrame(contrasts))
