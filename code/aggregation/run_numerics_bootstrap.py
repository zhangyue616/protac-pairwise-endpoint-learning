"""Read-only R5 numeric binding and the unchanged original component bootstrap.

No model imports, loading, fitting, queue entry, or prediction execution occurs.
"""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import math
import sys
import numpy as np
import pandas as pd

PROJECT = Path('/LOCAL_PROJECT_ROOT')
BASE = PROJECT/'intake/remaining_local_preparation_20261012/candidate_repair'
OLD = BASE/'r5_remaining_20261013'
CAND = BASE/'r5_dependency_preparation/candidate_r5_contract'
SAVED = OLD/'descriptive_summary'
OUT = Path(__file__).resolve().parent
ORIGINAL = PROJECT/'r5_analysis/analysis/summarize_r5.py'
DEP = CAND/'input_build/dependency/pair_dependency_components.csv'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()

def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)

def read(path):
    return json.loads(Path(path).read_text('utf-8'))

def csv(name, frame):
    frame.to_csv(OUT/name,index=False,lineterminator='\n')

def compare_frame(actual, saved, label):
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), saved.reset_index(drop=True),
        check_dtype=False, check_exact=False, check_names=False, rtol=2e-13, atol=2e-14)
    checks.append({'check':label,'status':'PASS','rows':len(actual)})

spec_mod = importlib.util.spec_from_file_location('original_r5_summary', ORIGINAL)
original = importlib.util.module_from_spec(spec_mod)
spec_mod.loader.exec_module(original)
require(original.DRAWS == 2000 and original.SEED == 20261027,'original bootstrap protocol drift')
checks=[]

def load_read_predictions():
    """Only redirect the historical hard-coded dependency path to its new input."""
    tree=ast.parse(ORIGINAL.read_text('utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='read_predictions')
    replacements=0
    for child in ast.walk(node):
        if (isinstance(child,ast.Call) and ast.unparse(child.func)=='pd.read_csv' and
            ast.unparse(child.args[0])=="root / 'input_build/dependency/pair_dependency_components.csv'"):
            child.args[0]=ast.Name(id='DEPENDENCY_PATH',ctx=ast.Load()); replacements+=1
    require(replacements==1,'dependency redirection must match exactly once')
    ns={**original.__dict__,'DEPENDENCY_PATH':DEP}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(ORIGINAL)+' [dependency path only]','exec'),ns)
    return ns['read_predictions']

def calibration_function():
    path=CAND/'engine/model_adapters.py'
    node=next(n for n in ast.parse(path.read_text('utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='nonnegative_l1_scale')
    ns={'np':np,'math':math,'Any':object}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path)+' [scale function only]','exec'),ns)
    return ns['nonnegative_l1_scale']

def selected_oof_audit(spec,pairs,selected):
    scale=calibration_function()
    inner=pd.read_csv(CAND/'input_build/config/inner_identity_folds.csv')
    outer=pd.read_csv(CAND/'input_build/data/outer_identity_folds.csv')
    y=pairs.canonical_target_b_minus_a.to_numpy(float)
    rows=[]
    for item in selected.itertuples():
        cell=PROJECT/Path(item.selection_path).parent
        selected_json=read(cell/'selection.json')['selected']
        path=cell/'candidates'/item.candidate_id/'oof_predictions.npz'
        with np.load(path,allow_pickle=False) as z:
            indices=z['pair_indices']; raw=z['raw_prediction']
        require(len(indices)==len(np.unique(indices)) and np.isfinite(raw).all(),'selected OOF invalid')
        assignment=inner.loc[inner.repeat.eq(item.repeat)&inner.outer_fold.eq(item.outer_fold)].set_index('identity_hash').inner_fold
        a=pairs.identity_hash_i.map(assignment);b=pairs.identity_hash_j.map(assignment)
        expected=np.flatnonzero((a.notna()&b.notna()&a.eq(b)).to_numpy())
        require(np.array_equal(indices,expected),'selected OOF identity/fold population mismatch')
        o=outer.loc[outer.repeat.eq(item.repeat)].set_index('identity_hash').outer_fold
        require(pairs.iloc[indices].identity_hash_i.map(o).ne(item.outer_fold).all() and pairs.iloc[indices].identity_hash_j.map(o).ne(item.outer_fold).all(),'outer-test identity in selected OOF')
        c=scale(y[indices],raw)
        raw_mae=float(np.mean(np.abs(y[indices]-raw)))
        cal_mae=float(np.mean(np.abs(y[indices]-c*raw)))
        require(abs(c-item.scale_c)<=1e-12 and abs(cal_mae-item.inner_oof_calibrated_mae)<=1e-12 and abs(raw_mae-selected_json['inner_oof_raw_mae'])<=1e-12,'selected OOF scale/MAE mismatch')
        rows.append({'representation':item.representation,'learner':item.learner,'repeat':item.repeat,'outer_fold':item.outer_fold,'pipeline':item.pipeline,'candidate_id':item.candidate_id,'n_oof':len(indices),'scale_saved':item.scale_c,'scale_recomputed':c,'calibrated_mae_recomputed':cal_mae,'status':'PASS'})
    csv('selected_oof_audit.csv',pd.DataFrame(rows))
    checks.append({'check':'all selected OOF population/calibration/MAE; no test identities','status':'PASS','rows':len(rows)})

def component_audit(pairs,pred):
    records=pd.read_csv(CAND/'input_build/dependency/record_dependency_components.csv')
    sources=pd.read_csv(CAND/'input_build/data/record_sources.csv')
    require(records.record_id.is_unique and set(records.record_id)==set(sources.record_id),'record dependency coverage')
    linked=records.merge(sources[['record_id','identity_hash','c2_source_id']],on='record_id',validate='1:1',suffixes=('','_source'))
    require(linked.identity_hash.eq(linked.identity_hash_source).all() and linked.c2_source_id.eq(linked.c2_source_id_source).all(),'record identity/source dependency mapping drift')
    # Independent connected partition check, without assuming historical labels/counts.
    parent={r:r for r in records.record_id}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[b]=a
    for key in ['identity_hash','c2_source_id']:
        for _,g in records.groupby(key,sort=False):
            ids=g.record_id.to_list()
            for item in ids[1:]:union(ids[0],item)
    records['rebuilt_root']=[find(x) for x in records.record_id]
    require(records.groupby('rebuilt_root').dependency_component.nunique().eq(1).all() and records.groupby('dependency_component').rebuilt_root.nunique().eq(1).all(),'source/shared-identity component partition mismatch')
    mapping=records.set_index('record_id').dependency_component
    dep=pd.read_csv(DEP).set_index('pair_id')
    require(set(dep.index)==set(pairs.pair_id),'all repaired pairs must have component')
    actual=pairs.record_id_i.map(mapping)
    require(actual.eq(pairs.record_id_j.map(mapping)).all(),'pair straddles dependency component')
    require(actual.to_numpy().tolist()==dep.reindex(pairs.pair_id).dependency_component.to_list(),'pair component assignment mismatch')
    require(pred.dependency_component.eq(pred.pair_id.map(dep.dependency_component)).all(),'prediction component assignment mismatch')
    support=pred.loc[pred.representation.eq('binary')&pred.learner.eq('ridge')&pred.pipeline.eq('direct_pair')]
    rows=[]
    for comp,g in support.groupby('dependency_component',sort=True):
        rows.append({'dependency_component':comp,'test_observations_across_repeats':len(g),'unique_test_pairs':g.pair_id.nunique(),**{f'R{r}':int(g.repeat.eq(r).sum()) for r in range(3)}})
    csv('bootstrap_component_support.csv',pd.DataFrame(rows))
    checks.append({'check':'independently rebuilt normalized-source/shared-identity partition','status':'PASS','rows':len(records)})
    return {'record_components':int(records.dependency_component.nunique()),'pair_components':int(dep.dependency_component.nunique()),'test_prediction_components':int(support.dependency_component.nunique())}

def source_audit(pairs,source_predictions):
    source=pd.read_csv(CAND/'input_build/data/record_sources.csv').set_index('record_id')
    truth=pairs.set_index('pair_id')
    require(not source_predictions.duplicated(['holdout_id','learner','pipeline','pair_id']).any(),'duplicate source prediction key')
    rows=[]
    for (hid,learner,arm),g in source_predictions.groupby(['holdout_id','learner','pipeline']):
        heldout=g.heldout_source.unique();require(len(heldout)==1,'source label inconsistent')
        test_records=set(source.index[source.c2_source_id.eq(heldout[0])])
        expected=set(pairs.loc[pairs.record_id_i.isin(test_records)&pairs.record_id_j.isin(test_records),'pair_id'])
        require(set(g.pair_id)==expected,'source heldout pair population mismatch')
        train_ids=set(source.loc[~source.c2_source_id.eq(heldout[0]),'identity_hash'])
        require(g.identity_unseen_i.eq(~g.identity_hash_i.isin(train_ids)).all() and g.identity_unseen_j.eq(~g.identity_hash_j.isin(train_ids)).all(),'source identity unseen mismatch')
        for field in ['record_id_i','record_id_j','identity_hash_i','identity_hash_j','normalized_poi_id','pair_context_hash']:
            require(g[field].to_list()==truth.reindex(g.pair_id)[field].to_list(),'source endpoint identity/context mismatch '+field)
        require(np.allclose(g.true_delta,truth.reindex(g.pair_id).canonical_target_b_minus_a,rtol=0,atol=1e-12),'source truth mismatch')
        rows.append({'holdout_id':hid,'learner':learner,'pipeline':arm,'n':len(g),'both_identity_unseen':int(g.both_identity_unseen.sum()),'status':'PASS'})
    csv('source_boundary_audit.csv',pd.DataFrame(rows))
    checks.append({'check':'all 33 source cells: heldout source, pair/identity/context/labels','status':'PASS','rows':len(rows)})

def ripk1_audit():
    root=OLD/'ripk1_case'
    members=pd.read_csv(root/'member_predictions.csv');ens=pd.read_csv(root/'ensemble_predictions.csv')
    require(not members.duplicated(['member_id','pair_id']).any(),'duplicate RIPK1 member key')
    require(members.member_id.nunique()==420 and len(members)==420*45,'RIPK1 member coverage')
    require(np.allclose(members.raw_prediction*members.scale_c,members.calibrated_prediction,rtol=1e-12,atol=1e-12),'RIPK1 member calibration')
    keys=['representation','learner','strategy','pair_id']
    means=members.groupby(keys)[['raw_prediction','calibrated_prediction']].mean()
    e=ens.set_index(keys).reindex(means.index)
    require(np.allclose(means,e[['raw_prediction','calibrated_prediction']],rtol=1e-12,atol=1e-12),'RIPK1 mean member-calibrated ensemble')
    require(members.groupby(keys).size().eq(15).all(),'RIPK1 15 members per pair')
    metric=pd.read_csv(root/'ensemble_metrics.csv');contrast=pd.read_csv(root/'contrasts.csv')
    mkeys=['representation','learner','strategy','layer','output_scale']
    for item in metric.itertuples():
        if item.representation=='reference':
            g=ens.loc[ens.representation.eq('binary')&ens.learner.eq('ridge')&ens.strategy.eq('direct_pair')]
        else:g=ens.loc[ens.representation.eq(item.representation)&ens.learner.eq(item.learner)&ens.strategy.eq(item.strategy)]
        if item.layer=='primary42':g=g.loc[g.primary_ge05]
        prediction=np.zeros(len(g)) if item.representation=='reference' else g['raw_prediction' if item.output_scale=='raw' else 'calibrated_prediction']
        m=original.metrics(g.true_delta,prediction)
        for saved_field,calc_field in [('n_pairs','n'),('n_direction','n_direction'),('MAE','mae'),('RMSE','rmse'),('strict_DA','DA_strict'),('half_zero_DA','DA_half'),('zero_predictions','zero_predictions'),('true_ties_excluded','true_ties_excluded')]:
            require(abs(getattr(item,saved_field)-m[calc_field])<=1e-12,'RIPK1 metric mismatch '+saved_field)
    lookup=metric.set_index(mkeys).MAE
    for item in contrast.itertuples():
        a=lookup[(item.representation,item.learner,item.lhs_strategy,item.layer,item.output_scale)]
        b=lookup[('reference','none','Zero',item.layer,item.output_scale)] if item.rhs_strategy=='Zero' else lookup[(item.representation,item.learner,item.rhs_strategy,item.layer,item.output_scale)]
        require(abs(a-b-item.MAE_difference)<=1e-12,'RIPK1 contrast sign/value mismatch')
    csv('ripk1_metrics_verified.csv',metric);csv('ripk1_contrasts_verified.csv',contrast)
    main=metric.loc[metric.layer.eq('primary42')&metric.output_scale.eq('calibrated')&~metric.representation.eq('reference')]
    table=main.pivot(index=['representation','learner'],columns='strategy',values='MAE').reset_index()
    table['E0matched_minus_E1']=table.endpoint_E0_matched-table.endpoint_E1
    table['Direct_minus_E1']=table.direct_pair-table.endpoint_E1
    table['Zero_MAE']=float(metric.loc[metric.layer.eq('primary42')&metric.output_scale.eq('calibrated')&metric.representation.eq('reference'),'MAE'].iloc[0])
    csv('table4_ripk1_primary42.csv',table)
    checks.append({'check':'RIPK1 420 saved members, ensemble, all metric/contrast cells','status':'PASS','rows':len(members)})
    return {'member_rows':len(members),'metric_rows':len(metric),'contrast_rows':len(contrast),**ripk1_key_stats(metric,contrast)}

def ripk1_key_stats(metric,contrast):
    result={}
    for layer in ['primary42','secondary45']:
        c=contrast.loc[contrast.layer.eq(layer)&contrast.output_scale.eq('calibrated')]
        matched=c.loc[c.contrast.eq('E0matched-E1'),'MAE_difference']
        direct=c.loc[c.contrast.eq('Direct-E1'),'MAE_difference']
        zero=c.loc[c.contrast.eq('Model-Zero')].groupby('strategy').MAE_difference.apply(lambda x:int(x.lt(0).sum())).to_dict()
        result[layer]={'matched_e1_better_cells':int(matched.gt(0).sum()),'matched_e0m_better_cells':int(matched.lt(0).sum()),'matched_difference_min':float(matched.min()),'matched_difference_max':float(matched.max()),'direct_lower_mae_than_e1_cells':int(direct.lt(0).sum()),'strategy_combinations_below_zero':zero,'zero_mae':float(metric.loc[metric.layer.eq(layer)&metric.output_scale.eq('calibrated')&metric.representation.eq('reference'),'MAE'].iloc[0])}
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    spec=read(CAND/'spec.json')
    spec['output_root']=(OLD/'merged_run_output').relative_to(PROJECT).as_posix()
    spec['inputs']={k:(CAND/v).relative_to(PROJECT).as_posix() for k,v in spec['inputs'].items()}
    spec['source_holdout']['selected_sources']=(OLD/'SOURCE_COMPLETION_SELECTION.json').relative_to(PROJECT).as_posix()
    pred,external,coverage,folds,selected,parameters,counts=load_read_predictions()(PROJECT,spec)
    print('Read all 420 saved outer units; auditing mappings and selected OOF arrays.',flush=True)
    compare_frame(pred,pd.read_csv(SAVED/'outer_predictions_long.csv'),'saved long table equals all 420 source cells')
    pairs=pd.read_csv(PROJECT/spec['inputs']['pairs'])
    pindex=pairs.set_index('pair_id')
    for field in ['record_id_i','record_id_j','identity_hash_i','identity_hash_j','normalized_poi_id','pair_context_hash']:
        require(pred[field].to_list()==pindex.reindex(pred.pair_id)[field].to_list(),'outer endpoint/context identity mismatch '+field)
    require(np.array_equal(pred.pair_row_index,pairs.reset_index().set_index('pair_id').reindex(pred.pair_id)['index']),'outer pair row index drift')
    components=component_audit(pairs,pred)
    selected_oof_audit(spec,pairs,selected)
    repeat,aggregate,primary,rep_compare,gamma,estimates=original.summarize_metrics(pred)
    for name,frame in [('edge_coverage.csv',coverage),('fold_counts.csv',folds),('repeat_metrics.csv',repeat),('learner_metrics.csv',aggregate),('primary_repeat_ratios.csv',primary),('representation_repeat_comparisons.csv',rep_compare),('representation_learner_comparisons.csv',gamma),('selected_candidates.csv',selected),('selection_parameter_boundaries.csv',parameters)]:
        compare_frame(frame,pd.read_csv(SAVED/name),'recomputed '+name)
    old_summary=read(SAVED/'SUMMARY.json')
    require(all(abs(value-old_summary['estimates'][key])<=1e-13 for key,value in estimates.items()),'saved point estimate mismatch')
    # Direct edge-weighted computation verifies all 84 primary cells separately.
    direct=[]
    for key,g in pred.groupby(['representation','learner','repeat','pipeline']):
        m=float(np.sum(np.abs(g.true_delta-g.calibrated_prediction).to_numpy(float),dtype=np.float64)/len(g))
        row=repeat.loc[repeat.representation.eq(key[0])&repeat.learner.eq(key[1])&repeat.repeat.eq(key[2])&repeat.pipeline.eq(key[3])&repeat.scale.eq('calibrated')&repeat.stratum.eq('all')]
        require(len(row)==1 and abs(m-row.mae.iloc[0])<=1e-14,'direct pooled MAE mismatch')
        direct.append({**dict(zip(['representation','learner','repeat','pipeline'],key)),'n':len(g),'mae_direct_sum':m,'status':'PASS'})
    csv('primary_direct_aggregation_audit.csv',pd.DataFrame(direct))
    checks.append({'check':'direct equal-edge pooled primary MAE independent sum','status':'PASS','rows':len(direct)})
    # Original function unchanged: common component draws, no epsilon/NA deletion.
    print('Numeric summaries match; executing unchanged original 2000-draw bootstrap.',flush=True)
    draws,intervals=original.bootstrap(pred,estimates)
    csv('conditional_component_bootstrap.csv',draws);csv('primary_estimates.csv',intervals)
    csv('table2_contrasts.csv',intervals.loc[~intervals.statistic.str.startswith('log_C1_over_direct')])
    csv('table_s8_contrasts.csv',intervals)
    compare=gamma[['learner','pipeline','mae_binary','mae_count','absolute_mae_difference']].copy()
    zero=repeat.loc[repeat.pipeline.eq('zero')&repeat.stratum.eq('all')]
    require(zero.groupby('repeat').mae.nunique().eq(1).all(),'Zero is not common')
    csv('table1_mae.csv',compare)
    csv('table_s6_primary_metrics.csv',repeat.loc[repeat.stratum.eq('all')])
    csv('table_s7_supplementary_strata.csv',repeat.loc[~repeat.stratum.eq('all')])
    csv('table_s6_zero.csv',zero.loc[zero.representation.eq('binary')&zero.learner.eq('ridge')])
    frequency=selected.groupby(['representation','learner','pipeline','candidate_id']).size().rename('outer_models').reset_index()
    boundaries=parameters.loc[parameters.varied_numeric_grid].groupby(['representation','learner','pipeline','parameter']).agg(outer_models=('candidate_id','size'),at_min=('at_numeric_min','sum'),at_max=('at_numeric_max','sum')).reset_index()
    compare_frame(frequency,pd.read_csv(SAVED/'selection_candidate_frequencies.csv'),'selection frequencies')
    compare_frame(boundaries,pd.read_csv(SAVED/'selection_boundary_summary.csv'),'selection endpoints')
    csv('table_s9_selection_boundaries.csv',boundaries)
    # Reuse source numerical implementation; only its administrative logical-fit count changes.
    tree=ast.parse(ORIGINAL.read_text('utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='summarize_sources')
    target="status['ledger_events'].get('FIT_COMPLETE')"
    edits=0
    class CensusOnly(ast.NodeTransformer):
        def visit_Call(self,node):
            nonlocal edits
            self.generic_visit(node)
            if ast.unparse(node)==target:
                edits+=1;return ast.parse("status['logical_fit_coverage']",mode='eval').body
            return node
    node=CensusOnly().visit(node);require(edits==1,'source administrative census must match once')
    ns={**original.__dict__};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(ORIGINAL)+' [logical fit census only]','exec'),ns)
    source_pred,source_metrics,source_ratios,source_selected,source_counts=ns['summarize_sources'](PROJECT,spec)
    source_audit(pairs,source_pred)
    for name,frame in [('source_metrics.csv',source_metrics),('source_descriptive_ratios.csv',source_ratios),('source_selected_candidates.csv',source_selected)]:
        compare_frame(frame,pd.read_csv(SAVED/name),'recomputed '+name);csv(name,frame)
    ensemble,external_metrics=original.summarize_external(external)
    compare_frame(ensemble,pd.read_csv(SAVED/'external_ensemble_predictions.csv'),'BRD4 saved member ensemble')
    compare_frame(external_metrics,pd.read_csv(SAVED/'external_metrics.csv'),'BRD4 saved metrics')
    csv('external_metrics.csv',external_metrics)
    ripk1=ripk1_audit()
    checks.append({'check':'bootstrap function from original module; fixed seed/draws/common weights/NA propagation','status':'PASS','rows':len(draws)})
    stats={
      'total_pairs':len(pairs),'test_observations':int(coverage.coverage_repeats.sum()),'unique_test_pairs':int(coverage.coverage_repeats.gt(0).sum()),
      'coverage_0_1_2_3':coverage.coverage_repeats.value_counts().sort_index().to_dict(),
      'repeat_test_observations':zero.loc[zero.representation.eq('binary')&zero.learner.eq('ridge'),['repeat','n','n_direction','true_ties_excluded','mae']].to_dict('records'),
      'matched_e1_lower_mae_cells':{rep:int(primary.loc[primary.representation.eq(rep),'difference_E0matched_minus_E1'].gt(0).sum()) for rep in original.REPS},
      'count_lower_mae_combinations':int(gamma.absolute_mae_difference.lt(0).sum()),
      'primary_mae_min':float(aggregate.loc[aggregate.scale.eq('calibrated')&aggregate.stratum.eq('all'),'mean_mae'].min()),
      'primary_mae_max':float(aggregate.loc[aggregate.scale.eq('calibrated')&aggregate.stratum.eq('all'),'mean_mae'].max()),
      'zero_mean_mae':float(zero.mae.mean()),'components':components,'ripk1':ripk1}
    stats['error_reduction_percent_min']=100*(1-stats['primary_mae_max']/stats['zero_mean_mae'])
    stats['error_reduction_percent_max']=100*(1-stats['primary_mae_min']/stats['zero_mean_mae'])
    stats['primary_combinations_below_zero']=int(aggregate.loc[aggregate.scale.eq('calibrated')&aggregate.stratum.eq('all'),'mean_mae'].lt(stats['zero_mean_mae']).sum())
    ext=external_metrics.loc[external_metrics.scale.eq('calibrated')]
    ext_b=ext.loc[ext.representation.eq('binary')].set_index(['learner','pipeline'])
    ext_c=ext.loc[ext.representation.eq('count')].set_index(['learner','pipeline']).reindex(ext_b.index)
    stats['brd4_count_vs_binary']={'lower_MAE':int(ext_c.mae.lt(ext_b.mae).sum()),'higher_strict_DA':int(ext_c.DA_strict.gt(ext_b.DA_strict).sum()),'higher_half_DA':int(ext_c.DA_half.gt(ext_b.DA_half).sum()),'total_combinations':len(ext_b)}
    stats['source_direct_vs_e1']=source_ratios.assign(E1_lower_MAE=source_ratios.log_Direct_over_E1.gt(0)).groupby(['holdout_id','scenario']).E1_lower_MAE.agg(['sum','count']).reset_index().to_dict('records')
    original.dump(OUT/'KEY_MANUSCRIPT_NUMBERS.json',stats)
    bindings={p.as_posix():digest(p) for p in [ORIGINAL,CAND/'spec.json',CAND/'engine/model_adapters.py',DEP,CAND/'input_build/dependency/record_dependency_components.csv',PROJECT/spec['inputs']['pairs'],PROJECT/spec['inputs']['outer_folds'],PROJECT/spec['inputs']['inner_folds'],SAVED/'outer_predictions_long.csv',SAVED/'SUMMARY.json',OLD/'PARENT_COMPLETION_MANIFEST.json',OLD/'ripk1_case/RESULTS_RECEIPT.json']}
    summary={'status':'EXECUTOR_NUMERICS_AND_ORIGINAL_BOOTSTRAP_COMPLETE_PENDING_SINGLE_STAGE_ACCEPTANCE','scientific_fits_added':0,'models_loaded':0,'prediction_calls':0,'original_implementation':ORIGINAL.as_posix(),'bootstrap_function_changed':False,'draws_requested':2000,'seed':20261027,'aggregation':'pair-weighted pool of five outer folds within repetition; log ratio in each learner/repetition cell; equal repetitions then equal eligible learners','component_weighting':'same component multiplicities for all representations, learners, repetitions and arms; pair-weighted MAE inside each cell','zero_mae_policy':'undefined ratios/empty required cells propagate NA to affected statistic; no epsilon; no silently omitted cells','uncertainty':'conditional fixed selected fitted predictions; original percentile bootstrap only, no total training/selection uncertainty or multiplicity adjustment','runtime':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__},'statistics':stats,'estimates':estimates,'intervals':intervals.to_dict('records'),'checks':checks,'input_bindings_sha256':bindings,'limitations':['All 420 selected-candidate OOF populations/scales/losses recomputed; candidate ranking checked against all eight saved candidate summaries, the 2940 unselected candidate OOF arrays were not re-evaluated.','Model training and atomic fit execution reused from prior administrative acceptance, not rerun or re-audited.','Source truth/assay certification, author approval, release and submission remain separate gates.'],'execution_authority':'EXECUTION_COMMISSION.md 2026-10-16; overrides historical bootstrap_authorized=false for this new read-only postprocessing directory only'}
    original.dump(OUT/'NUMERICS_BOOTSTRAP_SUMMARY.json',summary)
    csv('numerical_checks.csv',pd.DataFrame(checks))
    lines=['# R5 数值核对与原协议 bootstrap 执行结果','',
      '执行完成；供主控指定的同一次阶段接受使用，本报告不代替该接受。置信度：高（本报告覆盖的数值与绑定）。','',
      f"- 输入：{len(pairs)} pairs；{stats['test_observations']} test observations；{stats['unique_test_pairs']} unique tested pairs。",
      f"- 修复后映射：records {components['record_components']} components，pairs {components['pair_components']}，实际测试预测 {components['test_prediction_components']}；未照抄旧38。",
      '- 原函数 bootstrap 完全复用，2000 draws、seed 20261027；共同component权重，先合并折、再log、再等权重复/learner；不加epsilon、不静默丢格。',
      f"- 全部14统计量的有效draw为{int(intervals.valid_draws.min())}–{int(intervals.valid_draws.max())}；每项实际计数和95%条件区间见 primary_estimates.csv。",
      f"- E1较E0-matched低MAE：binary {stats['matched_e1_lower_mae_cells']['binary']}/9；count {stats['matched_e1_lower_mae_cells']['count']}/9。count较binary低MAE：{stats['count_lower_mae_combinations']}/14。",
      '- 核对：420保存外折单元、两表示/策略配对、真实标签/端点/上下文/身份、重复键、全部selected OOF的支持/标定/MAE、八候选保存评分的选参规则、84主单元直接误差求和、汇总/端点频次到表格。',
      '- 三来源33单元的source/identity边界、BRD4成员标定后平均与指标、RIPK1 420成员/42与45层及全部MAE对比方向均核对。',
      '- 输出：table1_mae.csv、table2_contrasts.csv、table_s6_primary_metrics.csv、table_s7_supplementary_strata.csv、table_s8_contrasts.csv、table_s9_selection_boundaries.csv；所有值保留全精度。',
      '- 适用范围：内部区间仅为固定已选拟合预测的component-resampling条件区间；LOCO/RIPK1不新增区间或p值。',
      '- 未加载模型、无fit/重训/队列恢复。排名依据全部八候选已保存评分；未遍历2940个非选中候选OOF原阵列。此前训练执行闭包复用；来源/作者/发布/投稿门未关闭。','',
      '输入路径与少量身份SHA见 NUMERICS_BOOTSTRAP_SUMMARY.json；原实现入口：'+ORIGINAL.as_posix(),'']
    (OUT/'NUMERICS_BOOTSTRAP_REPORT.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print(json.dumps(original.clean({'status':summary['status'],'statistics':stats,'intervals':intervals.to_dict('records'),'checks':len(checks)}),ensure_ascii=False,allow_nan=False))

if __name__=='__main__':main()
