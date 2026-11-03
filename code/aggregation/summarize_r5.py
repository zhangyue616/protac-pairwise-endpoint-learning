"""Summarize the frozen R5 run once; no fitting or model selection is performed.

Aggregation and paired component draws follow summarize_route_c.py; this version
uses the revised graph, adds the prespecified matched-support contrasts, and
propagates undefined log contrasts instead of adding an epsilon/dropping cells.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd

LEARNERS=['ridge','random_forest','xgboost']
REPS=['binary','count']
ARMS=['direct_pair','endpoint_E0','endpoint_E1','endpoint_E0_matched','context_pair_C1']
TOL=1e-12
DRAWS=2000
SEED=20261027

def require(ok,message):
    if not bool(ok):raise RuntimeError(message)

def read_json(path):return json.loads(Path(path).read_text('utf-8'))

def clean(obj):
    if isinstance(obj,dict):return {str(k):clean(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [clean(x) for x in obj]
    if isinstance(obj,np.generic):obj=obj.item()
    if isinstance(obj,float) and not np.isfinite(obj):return None
    return obj

def dump(path,obj):path.write_text(json.dumps(clean(obj),indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n')

def arms_for(learner):return [a for a in ARMS if learner!='ridge' or a!='context_pair_C1']

def log_ratio(a,b):
    return float(np.log(a/b)) if np.isfinite(a) and np.isfinite(b) and a>0 and b>0 else np.nan

def metrics(y,p):
    y,p=np.asarray(y,float),np.asarray(p,float)
    require(np.isfinite(y).all() and np.isfinite(p).all(),'nonfinite metric input')
    require(not ((np.abs(y)<=TOL)&(y!=0)).any(),'near-zero nonzero truth requires explicit direction policy resolution')
    nz=y!=0;pzero=np.abs(p[nz])<=TOL;correct=(np.sign(y[nz])==np.sign(p[nz]))&~pzero
    return {'n':len(y),'n_direction':int(nz.sum()),'mae':float(np.mean(np.abs(y-p))) if len(y) else None,
      'rmse':float(np.sqrt(np.mean((y-p)**2))) if len(y) else None,
      'DA_strict':float(correct.mean()) if nz.any() else None,
      'DA_half':float(np.mean(correct.astype(float)+.5*pzero)) if nz.any() else None,
      'zero_predictions':int((np.abs(p)<=TOL).sum()),'true_ties_excluded':int((~nz).sum())}

def estimates_from_mae(mae):
    """Each cell is retained. A missing/nonpositive MAE propagates NA."""
    result={}
    for arm in ARMS:
        learners=LEARNERS[1:] if arm=='context_pair_C1' else LEARNERS
        result['gamma_'+arm]=float(np.mean([np.mean([log_ratio(mae[('count',l,r,arm)],mae[('binary',l,r,arm)]) for r in range(3)]) for l in learners]))
    for rep in REPS:
        for label,num,den in [('theta','direct_pair','endpoint_E1'),('theta_match','endpoint_E0_matched','endpoint_E1'),('theta_support','endpoint_E0','endpoint_E0_matched')]:
            result[label+'_'+rep]=float(np.mean([np.mean([log_ratio(mae[(rep,l,r,num)],mae[(rep,l,r,den)]) for r in range(3)]) for l in LEARNERS]))
        result['log_C1_over_direct_'+rep]=float(np.mean([np.mean([log_ratio(mae[(rep,l,r,'context_pair_C1')],mae[(rep,l,r,'direct_pair')]) for r in range(3)]) for l in LEARNERS[1:]]))
    result['delta']=result['theta_count']-result['theta_binary']
    return result

def read_predictions(root,spec):
    run=root/spec['output_root'];inputs=spec['inputs']
    pairs=pd.read_csv(root/inputs['pairs']);folds=pd.read_csv(root/inputs['outer_folds'])
    dep=pd.read_csv(root/'input_build/dependency/pair_dependency_components.csv')
    require(pairs.pair_id.is_unique and dep.pair_id.is_unique and set(dep.pair_id)==set(pairs.pair_id),'pair/component scope mismatch')
    truth=pairs.set_index('pair_id').canonical_target_b_minus_a
    coverage=pairs[['pair_id']].copy();expected={};census=[]
    for r in range(3):
        assignment=folds.loc[folds.repeat.eq(r)].set_index('identity_hash').outer_fold
        a,b=pairs.identity_hash_i.map(assignment),pairs.identity_hash_j.map(assignment)
        require(a.notna().all() and b.notna().all(),'identity assignment absent')
        coverage[f'R{r}']=a.eq(b).astype(int)
        for fold in range(5):
            expected[(r,fold)]=set(pairs.loc[a.eq(fold)&b.eq(fold),'pair_id'])
            census.append({'repeat':r,'outer_fold':fold,'test_edges':len(expected[(r,fold)])})
    coverage['coverage_repeats']=coverage[[f'R{r}' for r in range(3)]].sum(axis=1)
    metadata=pairs[['pair_id','time_status','any_approximate_label']].merge(dep[['pair_id','dependency_component']],on='pair_id',validate='1:1')
    frames=[];external=[];selected=[];parameters=[];boundary=[];counts=[]
    for job in spec['jobs']:
        rep,l,r,f,arm=[job[k] for k in ['representation','learner','repeat','outer_fold','pipeline']]
        cell=run/rep/'results'/f'{l}_R{r}'/f'outer_{f}'/arm
        status=read_json(run/rep/'runtime'/f'{l}_R{r}'/f'outer_{f}_{arm}.json')
        require(status['status']=='JOB_COMPLETE','job not complete: '+str(cell))
        frame=pd.read_csv(cell/'outer_predictions.csv')
        require(frame.pair_id.is_unique and set(frame.pair_id)==expected[(r,f)],'outer prediction support mismatch')
        require(frame.learner.eq(l).all() and frame.repeat.eq(r).all() and frame.outer_fold.eq(f).all() and frame.pipeline.eq(arm).all(),'outer keys mismatch')
        require(np.allclose(truth.reindex(frame.pair_id),frame.true_delta,rtol=0,atol=TOL),'outer labels mismatch')
        require(np.isfinite(frame[['raw_prediction','calibrated_prediction','scale_c']]).all().all(),'nonfinite prediction')
        require(frame.scale_c.ge(0).all() and np.allclose(frame.raw_prediction*frame.scale_c,frame.calibrated_prediction,rtol=TOL,atol=TOL),'calibration mismatch')
        frame.insert(0,'representation',rep);frames.append(frame.merge(metadata,on='pair_id',validate='m:1'))
        ext=pd.read_csv(cell/'external_predictions.csv');ext.insert(0,'representation',rep)
        require(len(ext)==21 and ext.pair_id.is_unique,'exploratory case support mismatch');external.append(ext)
        selection=read_json(cell/'selection.json');choice=selection['selected'];candidates=selection['candidates']
        require(len(candidates)==8 and len({c['candidate_id'] for c in candidates})==8,'candidate population mismatch')
        best=min(c['inner_oof_calibrated_mae'] for c in candidates)
        expected_choice=min((c for c in candidates if c['inner_oof_calibrated_mae']<=best+TOL),key=lambda c:c['candidate_id'])
        require(choice==expected_choice,'frozen selection rule mismatch')
        base={**job,'candidate_id':choice['candidate_id']}
        selected.append({**base,'scale_c':choice['scale_c'],'inner_oof_calibrated_mae':choice['inner_oof_calibrated_mae'],
          'selected_params_json':json.dumps(choice['params'],sort_keys=True),'selection_path':cell.relative_to(root).as_posix()+'/selection.json'})
        for parameter,value in choice['params'].items():
            unique=set(c['params'][parameter] for c in candidates)
            varied=len(unique)>1 and all(isinstance(x,(int,float)) and not isinstance(x,bool) for x in unique)
            parameters.append({**base,'parameter':parameter,'selected_value':str(value),'grid_values_json':json.dumps(sorted(unique,key=str)),
              'varied_numeric_grid':varied,'at_numeric_min':value==min(unique) if varied else False,'at_numeric_max':value==max(unique) if varied else False,
              'scope':'numeric endpoints; all selections in a two-level grid are endpoints'})
        expected_fits=125 if arm=='endpoint_E1' and l!='ridge' else 25
        require(status['fit_count']==expected_fits,'job fit count mismatch');counts.append({'scope':'internal',**job,'successful_fits':status['fit_count']})
    pred=pd.concat(frames,ignore_index=True);external=pd.concat(external,ignore_index=True)
    keys=['representation','learner','repeat','outer_fold','pipeline','pair_id']
    require(not pred.duplicated(keys).any() and not external.duplicated(keys).any(),'duplicate prediction key')
    a=pred.loc[pred.representation.eq('binary')];b=pred.loc[pred.representation.eq('count')]
    paired=a.merge(b,on=keys[1:],suffixes=('_binary','_count'),validate='1:1')
    require(len(paired)==len(a)==len(b),'representation supports differ')
    for field in ['record_id_i','record_id_j','identity_hash_i','identity_hash_j','normalized_poi_id','pair_context_hash','g_seen_in_outer_train','dependency_component']:
        require(paired[field+'_binary'].eq(paired[field+'_count']).all(),'paired metadata mismatch: '+field)
    return pred,external,coverage,pd.DataFrame(census),pd.DataFrame(selected),pd.DataFrame(parameters),pd.DataFrame(counts)

def summarize_metrics(pred):
    rows=[]
    for (rep,l,r,arm),frame in pred.groupby(['representation','learner','repeat','pipeline'],sort=True):
        masks={'all':np.ones(len(frame),bool),'abs_delta_ge_1':np.abs(frame.true_delta.to_numpy())>=1-TOL,
          'g_seen':frame.g_seen_in_outer_train.to_numpy(bool),'g_unseen':~frame.g_seen_in_outer_train.to_numpy(bool),
          'known_fixed_time':frame.time_status.eq('same_recorded_fixed_time').to_numpy(),
          'time_unknown':frame.time_status.eq('recorded_context_time_unknown').to_numpy(),
          'common_time_window':frame.time_status.eq('common_reported_time_window').to_numpy(),
          'has_approximate_label':frame.any_approximate_label.to_numpy(bool)}
        for name,mask in masks.items():
            sub=frame.loc[mask];base={'representation':rep,'learner':l,'repeat':r,'pipeline':arm,'stratum':name}
            for scale,col in [('raw','raw_prediction'),('calibrated','calibrated_prediction')]:rows.append({**base,'scale':scale,**metrics(sub.true_delta,sub[col])})
            if arm=='direct_pair':rows.append({**base,'pipeline':'zero','scale':'raw',**metrics(sub.true_delta,np.zeros(len(sub)))})
    repeat=pd.DataFrame(rows)
    aggregate=repeat.groupby(['representation','learner','pipeline','scale','stratum']).agg(repeats_nonempty=('mae','count'),
       mean_mae=('mae','mean'),min_mae=('mae','min'),max_mae=('mae','max'),mean_rmse=('rmse','mean'),
       mean_DA_strict=('DA_strict','mean'),mean_DA_half=('DA_half','mean'),observations=('n','sum')).reset_index()
    primary=repeat.loc[repeat.stratum.eq('all')&repeat.scale.eq('calibrated')].pivot(index=['representation','learner','repeat'],columns='pipeline',values='mae')
    require(primary[[a for a in ARMS if a!='context_pair_C1']].notna().all().all(),'primary MAE incomplete')
    for label,n,d in [('log_direct_over_E1','direct_pair','endpoint_E1'),('theta_match','endpoint_E0_matched','endpoint_E1'),('theta_support','endpoint_E0','endpoint_E0_matched'),('log_C1_over_direct','context_pair_C1','direct_pair')]:
        primary[label]=[log_ratio(a,b) for a,b in zip(primary[n],primary[d])]
    primary['difference_E0matched_minus_E1']=primary.endpoint_E0_matched-primary.endpoint_E1
    primary['difference_E0_minus_E0matched']=primary.endpoint_E0-primary.endpoint_E0_matched
    mae={(r.representation,r.learner,int(r.repeat),r.pipeline):r.mae for r in repeat.loc[repeat.stratum.eq('all')&repeat.scale.eq('calibrated')].itertuples()}
    comparisons=[]
    for learner in LEARNERS:
        for arm in arms_for(learner):
            for r in range(3):
                b,c=mae[('binary',learner,r,arm)],mae[('count',learner,r,arm)]
                comparisons.append({'learner':learner,'pipeline':arm,'repeat':r,'mae_binary':b,'mae_count':c,
                  'log_count_over_binary':log_ratio(c,b),'absolute_mae_difference':c-b})
    comparison=pd.DataFrame(comparisons)
    gamma=comparison.groupby(['pipeline','learner']).agg(log_count_over_binary=('log_count_over_binary',lambda x:float(np.mean(x.to_numpy()))),
        mae_binary=('mae_binary','mean'),mae_count=('mae_count','mean'),absolute_mae_difference=('absolute_mae_difference','mean')).reset_index()
    return repeat,aggregate,primary.reset_index(),comparison,gamma,estimates_from_mae(mae)

def bootstrap(pred,estimates):
    components=sorted(pred.dependency_component.unique());K=len(components);require(K>=2,'insufficient components')
    matrices={}
    for key,frame in pred.groupby(['representation','learner','repeat','pipeline'],sort=True):
        tmp=frame[['dependency_component']].copy();tmp['loss']=abs(frame.true_delta-frame.calibrated_prediction)
        g=tmp.groupby('dependency_component').loss.agg(['sum','count']).reindex(components,fill_value=0)
        matrices[key]=(g['sum'].to_numpy(),g['count'].to_numpy())
    rng=np.random.default_rng(SEED);rows=[]
    for draw in range(DRAWS):
        weights=np.bincount(rng.integers(0,K,K),minlength=K);mae={}
        for key,(sums,counts) in matrices.items():
            denominator=float(weights@counts);mae[key]=float(weights@sums)/denominator if denominator else np.nan
        rows.append({'draw':draw,**estimates_from_mae(mae)})
    draws=pd.DataFrame(rows);intervals=[]
    for statistic,estimate in estimates.items():
        values=draws.loc[np.isfinite(draws[statistic]),statistic].to_numpy()
        lo,hi=map(float,np.percentile(values,[2.5,97.5])) if len(values) else (None,None)
        intervals.append({'statistic':statistic,'estimate':estimate,'lower_95':lo,'upper_95':hi,'draws_requested':DRAWS,
          'valid_draws':len(values),'invalid_draws':DRAWS-len(values),'component_count':K,'seed':SEED,
          'interval_scope':'conditional on fixed selected fitted predictions; whole source/shared-identity components; no total uncertainty or multiplicity adjustment'})
    return draws,pd.DataFrame(intervals)

def summarize_external(members):
    require(not members.duplicated(['representation','learner','pipeline','repeat','outer_fold','pair_id']).any(),'external member key duplicated')
    require(np.allclose(members.raw_prediction*members.scale_c,members.calibrated_prediction,rtol=TOL,atol=TOL),'external scale mismatch')
    keys=['representation','learner','pipeline','pair_id'];rows=[]
    for key,g in members.groupby(keys,sort=True):
        require(len(g)==15 and set(zip(g.repeat,g.outer_fold))=={(r,f) for r in range(3) for f in range(5)},'external member set incomplete')
        require(g.true_delta.nunique()==1,'external label drift')
        rows.append({**dict(zip(keys,key)),'record_id_i':g.record_id_i.iloc[0],'record_id_j':g.record_id_j.iloc[0],
          'true_delta':g.true_delta.iloc[0],'raw_prediction':g.raw_prediction.mean(),'calibrated_prediction':g.calibrated_prediction.mean(),
          'members':15,'evidence_level':'previously inspected exploratory single series'})
    ensemble=pd.DataFrame(rows);metric=[]
    for key,g in ensemble.groupby(['representation','learner','pipeline']):
        for scale,col in [('raw','raw_prediction'),('calibrated','calibrated_prediction')]:
            metric.append({**dict(zip(['representation','learner','pipeline'],key)),'scale':scale,**metrics(g.true_delta,g[col])})
        if key[2]=='direct_pair':metric.append({'representation':key[0],'learner':key[1],'pipeline':'zero','scale':'raw',**metrics(g.true_delta,np.zeros(len(g)))})
    return ensemble,pd.DataFrame(metric)

def summarize_sources(root,spec):
    source=root/spec['output_root']/'source_holdout';selection=read_json(root/spec['source_holdout']['selected_sources']);rows=[];frames=[];counts=[];choices=[]
    for learner in LEARNERS:
        status=read_json(source/'runtime'/f'{learner}_status.json')
        require(status['status']=='COMPLETE','source learner incomplete')
        expected=225 if learner=='ridge' else 600
        require(status['ledger_events'].get('FIT_COMPLETE')==expected,'source fit count mismatch')
        counts.append({'scope':'source_holdout','representation':'binary','learner':learner,'successful_fits':expected})
        for src in selection['source_holdouts']:
            for arm in [a for a in arms_for(learner) if a!='endpoint_E0_matched']:
                cell=source/'results'/learner/src['holdout_id']/arm;f=pd.read_csv(cell/'source_predictions.csv');frames.append(f)
                require(len(f)==src['test_edges'] and int(f.both_identity_unseen.sum())==src['strict_identity_unseen_edges'],'source scenario support differs from frozen input')
                require(np.allclose(f.raw_prediction*f.scale_c,f.calibrated_prediction,rtol=TOL,atol=TOL),'source calibration mismatch')
                selected=read_json(cell/'selection.json');c=selected['selected'];candidates=selected['candidates']
                best=min(x['inner_oof_calibrated_mae'] for x in candidates)
                require(c==min((x for x in candidates if x['inner_oof_calibrated_mae']<=best+TOL),key=lambda x:x['candidate_id']),'source selection mismatch')
                choices.append({'learner':learner,'pipeline':arm,'holdout_id':src['holdout_id'],'source_index':src['source_index'],
                  'candidate_id':c['candidate_id'],'scale_c':c['scale_c'],'params_json':json.dumps(c['params'],sort_keys=True)})
                for scenario,mask in [('source_unseen',np.ones(len(f),bool)),('source_identity_unseen',f.both_identity_unseen.to_numpy(bool)),
                     ('one_identity_seen',f.identity_scenario.eq('one_seen').to_numpy()),('both_identity_seen',f.identity_scenario.eq('both_seen').to_numpy())]:
                    sub=f.loc[mask]
                    for scale,col in [('raw','raw_prediction'),('calibrated','calibrated_prediction')]:
                        rows.append({'learner':learner,'pipeline':arm,'holdout_id':src['holdout_id'],'source_index':src['source_index'],'scenario':scenario,'scale':scale,**metrics(sub.true_delta,sub[col])})
                    if arm=='direct_pair':rows.append({'learner':learner,'pipeline':'zero','holdout_id':src['holdout_id'],'source_index':src['source_index'],'scenario':scenario,'scale':'raw',**metrics(sub.true_delta,np.zeros(len(sub)))})
    table=pd.DataFrame(rows);ratios=[]
    for scenario in ['source_unseen','source_identity_unseen']:
        p=table.loc[table.scenario.eq(scenario)&table.scale.eq('calibrated')].pivot(index=['holdout_id','learner'],columns='pipeline',values='mae')
        for (holdout,learner),r in p.iterrows():ratios.append({'holdout_id':holdout,'learner':learner,'scenario':scenario,
           'Direct_MAE':r.direct_pair,'E0_MAE':r.endpoint_E0,'E1_MAE':r.endpoint_E1,'log_Direct_over_E1':log_ratio(r.direct_pair,r.endpoint_E1)})
    return pd.concat(frames,ignore_index=True),table,pd.DataFrame(ratios),pd.DataFrame(choices),pd.DataFrame(counts)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--bundle',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    root=a.bundle.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(root));from contract import load_for_execution
    _,spec,cfg,_,_,schema=load_for_execution(root/'spec.json')
    require(read_json(root/'QUEUE_STATUS.json')['status']=='COMPLETE','real training queue is incomplete')
    pred,members,coverage,folds,selections,parameters,count=read_predictions(root,spec)
    repeat,aggregate,primary,comparisons,gamma,estimates=summarize_metrics(pred);draws,intervals=bootstrap(pred,estimates)
    ensemble,external_metrics=summarize_external(members)
    source_pred,source_metrics,source_ratios,source_selected,source_count=summarize_sources(root,spec)
    fit_count=pd.concat([count,source_count],ignore_index=True)
    require(int(fit_count.successful_fits.sum())==spec['expected_successful_fits'],'total successful fit count mismatch')
    ledger_events={}
    run=root/spec['output_root']
    for path in list(run.glob('*/runtime/logs/**/*.jsonl'))+list((run/'source_holdout/logs').glob('**/*.jsonl')):
        for line in path.read_text('utf-8').splitlines():
            event=json.loads(line)['event'];ledger_events[event]=ledger_events.get(event,0)+1
    require(ledger_events.get('FIT_COMPLETE')==17925,'actual ledger successful count mismatch')
    frequency=selections.groupby(['representation','learner','pipeline','candidate_id']).size().rename('outer_models').reset_index()
    boundary=parameters.loc[parameters.varied_numeric_grid].groupby(['representation','learner','pipeline','parameter']).agg(
        outer_models=('candidate_id','size'),at_min=('at_numeric_min','sum'),at_max=('at_numeric_max','sum')).reset_index()
    outputs={'outer_predictions_long.csv':pred,'edge_coverage.csv':coverage,'fold_counts.csv':folds,'repeat_metrics.csv':repeat,
      'learner_metrics.csv':aggregate,'primary_repeat_ratios.csv':primary,'representation_repeat_comparisons.csv':comparisons,
      'representation_learner_comparisons.csv':gamma,'conditional_component_bootstrap.csv':draws,'primary_estimates.csv':intervals,
      'selected_candidates.csv':selections,'selection_parameter_boundaries.csv':parameters,'selection_candidate_frequencies.csv':frequency,
      'selection_boundary_summary.csv':boundary,'external_per_model_predictions.csv':members,'external_ensemble_predictions.csv':ensemble,
      'external_metrics.csv':external_metrics,'source_predictions.csv':source_pred,'source_metrics.csv':source_metrics,
      'source_descriptive_ratios.csv':source_ratios,'source_selected_candidates.csv':source_selected,'fit_accounting.csv':fit_count}
    for name,frame in outputs.items():frame.to_csv(out/name,index=False,lineterminator='\n')
    summary={'status':'COMPLETE_FROZEN_R5_SUMMARY','scientific_fits_added_by_analysis':0,'run_id':spec['run_id'],'input_schema':schema,
      'unique_test_pairs':int(coverage.coverage_repeats.gt(0).sum()),'test_pair_observations':int(coverage.coverage_repeats.sum()),
      'coverage_0_1_2_3':coverage.coverage_repeats.value_counts().sort_index().to_dict(),'actual_prediction_components':int(pred.dependency_component.nunique()),
      'aggregation':'pair weighted pool across five folds within repeat; log; equal repeats and learners','estimates':estimates,
      'intervals':intervals.to_dict('records'),'zero_mae_policy':'log NA; no epsilon; no silent cell deletion; retain absolute metrics/differences',
      'expected_successful_fits':spec['expected_successful_fits'],'ledger_events':ledger_events,'internal_models':len(spec['jobs']),'source_models':33,
      'new_external_admitted_sources':0,'new_external_evaluation_performed':False,'old_external_case':{'records':7,'pairs':21,'arms':28,'members_each':15,'exploratory':True},
      'source_holdouts':['C2_S0','C2_S1','C2_S3'],'source_inference':'descriptive per-source; no tiny-source population CI',
      'uncertainty':'conditional fixed selected fitted predictions, 2000 paired draws from revised source/shared identity components, seed20261027; no total uncertainty/multiplicity adjustment/equivalence test',
      'acceptance':'numeric integration only; final technical/independent scientific and author review separate'}
    dump(out/'SUMMARY.json',summary)
    print(json.dumps(clean({'status':summary['status'],'estimates':estimates,'actual_prediction_components':summary['actual_prediction_components'],'ledger_events':ledger_events})))

if __name__=='__main__':main()
