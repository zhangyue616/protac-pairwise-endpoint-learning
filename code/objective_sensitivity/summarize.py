"""Frozen finite-population readouts; repeated errors, never averaged predictions."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

KEYS = ['domain','representation','pipeline','objective']

def summarize(predictions, common_ids, out):
    out = Path(out)
    def require(condition, message):
        if not condition:
            raise RuntimeError(message)
    data = predictions.copy()
    require(np.isfinite(data[['true_delta','raw_prediction','scale_c','calibrated_prediction','absolute_error']].to_numpy()).all(), 'Nonfinite output')
    require(not data.duplicated(['slot_id','objective','pair_id']).any(), 'Duplicate heldout prediction')
    identities = data.groupby('pair_id',sort=True).agg(y_min=('true_delta','min'), y_max=('true_delta','max'), components=('component_id','nunique'))
    require(((identities.y_max-identities.y_min).abs() <= 1e-12).all() and (identities.components == 1).all(), 'Cross-domain pair identity/target conflict')
    counts = []
    for keys, block in data.groupby(KEYS, sort=True):
        domain = keys[0]
        expected_obs, expected_unique = (3450,2825) if domain == 'internal' else (5804,5804)
        require(len(block) == expected_obs and block.pair_id.nunique() == expected_unique, 'Output population incomplete')
        repetitions = block.groupby('pair_id').size()
        counts.append(dict(zip(KEYS,keys)) | {'observations':len(block),'unique_pairs':int(block.pair_id.nunique()),'components':int(block.component_id.nunique()),'outer_folds':len(block[['repeat','outer_fold']].drop_duplicates()),'repeat_count':int(block.repeat.nunique()),'multiplicity_histogram':json.dumps({int(k):int(v) for k,v in repetitions.value_counts().sort_index().items()})})
    # Average absolute error within pair before either pair or component weighting.
    pair_errors = data.groupby(KEYS+['pair_id','component_id'],as_index=False,sort=True).agg(absolute_error=('absolute_error','mean'), observation_count=('absolute_error','size'))
    metrics, component_rows = [], []
    for population in ('native_observations','unique_pairs','common_pairs'):
        if population == 'native_observations':
            sample = data[KEYS+['pair_id','component_id','absolute_error']].copy()
        else:
            sample = pair_errors
            if population == 'common_pairs':
                sample = sample[sample.pair_id.isin(common_ids)]
        for keys, block in sample.groupby(KEYS,sort=True):
            base = dict(zip(KEYS,keys)) | {'population':population,'evaluation_rows':len(block),'unique_pairs':int(block.pair_id.nunique()),'components':int(block.component_id.nunique())}
            components = block.groupby('component_id',sort=True).absolute_error.mean()
            for weighting, value in (('pair_micro',block.absolute_error.mean()),('component_macro',components.mean())):
                metrics.append(base | {'weighting':weighting,'mae':float(value)})
            for group, value in components.items():
                group_rows = block[block.component_id == group]
                component_rows.append(base | {'component_id':group,'component_mae':float(value),'component_rows':len(group_rows),'component_unique_pairs':int(group_rows.pair_id.nunique())})
    metrics = pd.DataFrame(metrics)
    groupkeys = ['domain','representation','objective','population','weighting']
    contrasts = metrics.pivot(index=groupkeys,columns='pipeline',values='mae').reset_index()
    require({'endpoint_E1','endpoint_E0_matched'}.issubset(contrasts.columns), 'Matched pipeline missing')
    contrasts['delta_E1_minus_E0matched'] = contrasts.endpoint_E1-contrasts.endpoint_E0_matched
    sensitivity = contrasts.pivot(index=['domain','representation','population','weighting'],columns='objective',values='delta_E1_minus_E0matched').reset_index()
    sensitivity['delta_change_macro_objective_minus_micro_objective'] = sensitivity.component_macro-sensitivity.pair_micro
    components = pd.DataFrame(component_rows)
    ckeys = ['domain','representation','objective','population','component_id']
    ccontrasts = components.pivot(index=ckeys,columns='pipeline',values='component_mae').reset_index()
    ccontrasts['delta_E1_minus_E0matched'] = ccontrasts.endpoint_E1-ccontrasts.endpoint_E0_matched
    csensitivity = ccontrasts.pivot(index=['domain','representation','population','component_id'],columns='objective',values='delta_E1_minus_E0matched').reset_index()
    csensitivity['delta_change_macro_objective_minus_micro_objective'] = csensitivity.component_macro-csensitivity.pair_micro
    for name, frame in [('POPULATION_COUNTS',pd.DataFrame(counts)),('PAIR_MEAN_ABSOLUTE_ERRORS',pair_errors),('METRICS',metrics),('MATCHED_CONTRASTS',contrasts),('OBJECTIVE_SENSITIVITY',sensitivity),('COMPONENT_METRICS',components),('COMPONENT_CONTRASTS',ccontrasts),('COMPONENT_OBJECTIVE_SENSITIVITY',csensitivity)]:
        frame.to_csv(out/(name+'.csv'),index=False,lineterminator='\n')
    return {'prediction_rows':len(data),'unique_pair_error_rows':len(pair_errors),'common_pair_count':len(common_ids),'primary_unique_pair_contrasts':contrasts[contrasts.population == 'unique_pairs'].to_dict('records'),'primary_objective_sensitivity':sensitivity[sensitivity.population == 'unique_pairs'].to_dict('records'),'interpretation':'Posthoc fixed Ridge/grid/population sensitivity; no causal decomposition, equivalence or prospective validation claim.'}
