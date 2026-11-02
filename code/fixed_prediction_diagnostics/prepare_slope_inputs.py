"""Join released no-label predictions to locally reconstructed canonical labels."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',required=True,type=Path)
    parser.add_argument('--pairs',required=True,type=Path,help='Locally reconstructed pairs_revised.csv')
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    repo=args.repo_root.resolve()
    out=args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory must be empty')
    pairs=pd.read_csv(args.pairs)
    if len(pairs)!=5804 or not pairs.pair_id.is_unique or 'canonical_target_b_minus_a' not in pairs:
        raise ValueError('Expected 5804 unique reconstructed pairs with canonical_target_b_minus_a')
    labels=pairs.set_index('pair_id').canonical_target_b_minus_a
    if not np.isfinite(labels.to_numpy()).all():
        raise ValueError('Nonfinite reconstructed canonical label')
    prepared=[]
    for protocol,filename,component,expected_n,expected_k in [
        ('internal','S06_outer_predictions_long.csv.gz','dependency_component',3450,38),
        ('loco','S29_all_corrected_source_predictions.csv.gz','heldout_component_id',5804,43),
    ]:
        frame=pd.read_csv(repo/'results/predictions/current_companion'/filename)
        frame=frame.loc[frame.learner.eq('xgboost') & frame.pipeline.eq('endpoint_E1')].copy()
        frame['true_delta']=frame.pair_id.map(labels)
        for representation in ('binary','count'):
            selected=frame.loc[frame.representation.eq(representation)]
            if len(selected)!=expected_n or selected[component].nunique()!=expected_k:
                raise ValueError('Prediction population differs from fixed slope contract')
        if not np.isfinite(frame[['true_delta','calibrated_prediction']].to_numpy()).all():
            raise ValueError('Missing/nonfinite labels or predictions; no row is excluded')
        for endpoint in ('record_id_i','record_id_j'):
            if endpoint in frame:
                authority=pairs.set_index('pair_id')[endpoint]
                if not np.array_equal(frame[endpoint].to_numpy(str),frame.pair_id.map(authority).to_numpy(str)):
                    raise ValueError('Prediction endpoint direction differs from reconstructed pair authority')
        prepared.append((protocol,frame[['representation','true_delta','calibrated_prediction',component]]))
    baseline=pd.read_csv(repo/'results/conditional_slopes/FOUR_SLOPE_TABLE.csv')
    if len(baseline)!=4 or set(zip(baseline.protocol,baseline.representation))!={(p,r) for p in ('internal','loco') for r in ('binary','count')}:
        raise ValueError('Incomplete frozen four-slope baseline')
    (out/'inputs').mkdir(parents=True,exist_ok=True)
    for protocol,frame in prepared:
        frame.to_csv(out/'inputs'/(protocol+'_XGB_E1_predictions.csv'),index=False,lineterminator='\n')
    panels=[{'protocol':r.protocol,'representation':r.representation,'slope_b':float(r.slope)} for r in baseline.itertuples()]
    (out/'OLS_PARAMETERS.json').write_text(json.dumps({'panels':panels},indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'status':'LOCAL_LABELS_ATTACHED_TO_RELEASED_FIXED_PREDICTIONS','panels':4,
                      'internal_observations_per_representation':3450,'loco_pairs_per_representation':5804,
                      'fits':0,'predictions':0,'resampling_batches':0}))

if __name__=='__main__':
    main()
