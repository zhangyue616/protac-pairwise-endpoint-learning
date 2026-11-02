"""Run one frozen source-component/representation/learner/pipeline cell."""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '4'
import argparse
import json
from pathlib import Path
import sys
from types import FunctionType
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import model_adapters as adapters
from inner_objective import evaluate_inner_oof, select_candidate
import scientific_components as science

PIPES = ('direct_pair','endpoint_E0','endpoint_E0_matched','endpoint_E1')

def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8', newline='\n')

def configured_device(cfg, learner):
    device = cfg['learners'][learner]['device']
    if learner == 'xgboost':
        science.require(device in ('cuda', 'cuda:0'), 'Frozen XGBoost device must be CUDA ordinal 0')
        return 'cuda:0'
    science.require(device == 'cpu', 'Frozen Ridge/random-forest device must be CPU')
    return device

def require_actual_device(learner, device, receipt):
    if learner == 'xgboost':
        science.require(device == 'cuda:0' and receipt.get('actual_device') == device,
                        'XGBoost CPU fallback or unknown/wrong CUDA ordinal: '+str(receipt))

def fit(bundle, rep, pipeline, layer, learner, params, seed, device):
    science.require(device == ('cuda:0' if learner == 'xgboost' else 'cpu'), 'Frozen learner device differs')
    X, z, blocks = science.training(bundle, rep, pipeline, layer)
    if pipeline == 'endpoint_E1' and learner == 'ridge':
        model, receipt = adapters.fit_ridge_e1_within(float(params['alpha']), X, z, blocks)
        return model, {'fit_count':1, 'receipt':receipt}
    if pipeline == 'endpoint_E1':
        def checked_fit(name, fit_params, X_array, target, fit_seed, fit_device):
            science.require(name == learner and dict(fit_params) == params and fit_seed == seed and fit_device == device,
                            'E1 fit recipe or device differs')
            model, receipt = adapters.fit_model(name, fit_params, X_array, target, fit_seed, fit_device)
            require_actual_device(name, fit_device, receipt)
            return model, receipt
        # Bind the unchanged original five-round function body to a local
        # fit hook, so fallback fails before its training prediction/next round.
        original = adapters.fit_tree_e1_backfit
        namespace = dict(vars(adapters), fit_model=checked_fit)
        backfit = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
        result = backfit(learner, params, X, z, blocks, seed, device, iterations=5)
        science.require(result['total_fit_count'] == 5 and len(result['iteration_receipts']) == 5,
                        'E1 must retain exactly five fit rounds')
        return result['model'], {key:value for key,value in result.items() if key != 'model'}
    effective = dict(params)
    if learner == 'ridge':
        effective['fit_intercept'] = pipeline != 'direct_pair'
    model, receipt = adapters.fit_model(learner, effective, X, z, seed, device)
    require_actual_device(learner, device, receipt)
    return model, {'fit_count':1, 'receipt':receipt}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--component', required=True)
    parser.add_argument('--learner', required=True, choices=('ridge','random_forest','xgboost'))
    parser.add_argument('--representation', required=True, choices=('binary','count'))
    parser.add_argument('--pipeline', required=True, choices=PIPES)
    parser.add_argument('--check-inputs-only', action='store_true', help='Build and validate matrices without fit or prediction')
    args = parser.parse_args()
    cfg = science.js(Path(__file__).with_name('scientific_config.json'))
    device = configured_device(cfg, args.learner)
    science.require(args.component in cfg['outer_component_ids'], 'Unknown component')
    science.HELDOUT_COMPONENT = args.component
    bundle = science.bundle(args.data_dir.resolve(), True, science.map_counts)
    science.require(np.sum(bundle.pg == args.component) > 0, 'Component has no eligible test pairs')
    train_pair_ids = bundle.pairs.pair_id.to_numpy()[bundle.pg != args.component]
    science.require(len(set(bundle.pg[bundle.pg != args.component])) == 42, 'Expected 42 training pair-positive components')
    if args.check_inputs_only:
        print(json.dumps({'status':'INPUTS_VALIDATED_NO_FIT_NO_PREDICTION','records':len(bundle.records),
                          'pairs':len(bundle.pairs),'test_pairs':int(np.sum(bundle.pg == args.component)),
                          'layers':len(bundle.layers),'fits':0,'predictions':0}))
        return
    from portable import check_versions
    check_versions(science.js(Path(__file__).resolve().parents[1]/'core/scientific_config.json'))
    out = args.output_dir.resolve()
    science.require(not out.exists() or not any(out.iterdir()), 'Output must be a new or empty directory')
    out.mkdir(parents=True, exist_ok=True)
    candidates, summaries = science.candidate_grid(cfg, args.learner), []
    with threadpool_limits(limits=4):
        for candidate in candidates:
            pieces = []
            for layer in (0,1,2):
                model, receipt = fit(bundle,args.representation,args.pipeline,layer,args.learner,candidate['params'],cfg['model_seed'],device)
                indices = science.masks(bundle,layer)[3]
                raw = science.predict(bundle,args.representation,args.pipeline,layer,model,indices,science.predict_pipeline)
                pieces.append(pd.DataFrame({'pair_id':bundle.pairs.pair_id.to_numpy()[indices],
                                           'component_id':bundle.pg[indices],'true_delta':bundle.y[indices],
                                           'raw_prediction':raw}))
                save_json(out/'fit_receipts'/candidate['candidate_id']/f'inner_{layer}.json',receipt)
            oof = pd.concat(pieces,ignore_index=True).sort_values('pair_id',kind='mergesort').reset_index(drop=True)
            science.require(oof.pair_id.is_unique and set(oof.pair_id) == set(train_pair_ids), 'Incomplete/duplicate OOF pairs')
            score = evaluate_inner_oof(oof.true_delta.to_numpy(),oof.raw_prediction.to_numpy(),oof.component_id.to_numpy(),
                                       'component_macro',expected_components=42)
            summaries.append({**candidate,**score})
            path = out/'oof'/args.representation/args.pipeline/(candidate['candidate_id']+'.csv')
            path.parent.mkdir(parents=True,exist_ok=True)
            oof.to_csv(path,index=False,lineterminator='\n')
        selected = select_candidate(summaries)
        save_json(out/'selection'/args.representation/(args.pipeline+'.json'), {'selected':selected,'candidates':summaries})
        model, receipt = fit(bundle,args.representation,args.pipeline,-1,args.learner,selected['params'],cfg['model_seed'],device)
        indices = science.masks(bundle,-1)[3]
        raw = science.predict(bundle,args.representation,args.pipeline,-1,model,indices,science.predict_pipeline)
        calibrated = selected['scale_c']*raw
        result = pd.DataFrame({'pair_id':bundle.pairs.pair_id.to_numpy()[indices],
                               'heldout_component_id':args.component,'component_id':bundle.pg[indices],
                               'representation':args.representation,'learner':args.learner,'pipeline':args.pipeline,
                               'true_delta':bundle.y[indices],'raw_prediction':raw,
                               'scale_c':selected['scale_c'],'calibrated_prediction':calibrated,
                               'selected_candidate_id':selected['candidate_id']})
        result.to_csv(out/'OUTER_PREDICTIONS.csv',index=False,lineterminator='\n')
        save_json(out/'outer_fit_receipt.json',receipt)
        joblib.dump(model,out/'selected_model.joblib',compress=1)
    save_json(out/'RUN_SUMMARY.json',{'component':args.component,'representation':args.representation,
                                    'learner':args.learner,'pipeline':args.pipeline,'seed':cfg['model_seed'],'device':device,
                                    'candidate_count':8,'inner_folds':3,'test_pairs':len(indices),
                                    'expected_fit_calls':25*(5 if args.pipeline=='endpoint_E1' and args.learner!='ridge' else 1),
                                    'inner_objective':'component_macro','cache_reuse':False})
    print(json.dumps({'status':'COMPLETE','test_pairs':len(indices)}))

if __name__ == '__main__':
    main()
