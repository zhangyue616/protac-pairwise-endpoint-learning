"""Thin current-input adaptation of the existing R5 SI table formatter.

Formatting and prose-number integration only; consumes the single accepted
stage-one outputs and the separately accepted R6 corrected_shadow results.
"""
from pathlib import Path
import csv
import importlib.util
import json
import re
import shutil
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.dont_write_bytecode=True
PROJECT=Path('/LOCAL_PROJECT_ROOT')
INTAKE=PROJECT/'intake'
R3=INTAKE/'revision_r3'
WORK=R3/'tables_source_build'
BASE=INTAKE/'remaining_local_preparation_20261012/candidate_repair'
POST=BASE/'r5_postprocessing_20261016'
NUM=POST/'numerics_bootstrap'
DIAG=POST/'diagnostics'
MAP=POST/'evidence_mapping'
OLD=BASE/'r5_remaining_20261013'
CAND=BASE/'r5_dependency_preparation/candidate_r5_contract'
LEGACY=PROJECT/'r5_analysis'
PRIOR=INTAKE/'revision_r1/source/Supporting_Information.md'
SAVED=OLD/'descriptive_summary'
DESC=WORK/'current_inputs/data_description'
SUMMARY=WORK/'current_inputs/summary'
COMP=R3/'si_companions'
L={'ridge':'Ridge','random_forest':'RF','xgboost':'XGB'}
A={'direct_pair':'Direct','endpoint_E0':'E0','endpoint_E0_matched':'E0-matched','endpoint_E1':'E1','context_pair_C1':'C1','zero':'Zero'}
R={'binary':'Binary','count':'Count'}

def read(p):return json.loads(Path(p).read_text('utf-8'))
def write(p,t):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(t,encoding='utf-8',newline='\n')
def dump(p,x):write(p,json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def load(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def require(x,m):
    if not bool(x):raise RuntimeError(m)
def replace_once(t,old,new):
    require(t.count(old)==1,'exact replacement count is not one: '+old[:100]);return t.replace(old,new,1)
def fmt(x,n=6,signed=False):
    if pd.isna(x):return '—'
    return (f'{float(x):+.{n}f}' if signed else f'{float(x):.{n}f}').replace('-','−')
def md(headers,rows):
    esc=lambda x:str(x).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+['| '+' | '.join(esc(x) for x in row)+' |' for row in rows])
def table(sid,title,caption,headers,rows):
    return f'### Table S{sid}. {title}\n\n{caption}\n\n'+md(headers,rows)+'\n'
def section(t,sid):
    m=re.search(rf'^### Table S{sid}\. .*?(?=^### Table S\d+\.|^## Supplementary Texts|\Z)',t,re.M|re.S);require(m is not None,f'S{sid} absent');return m.group().rstrip()

def prepare_data():
    require(read(POST/'stage1_acceptance/FINAL_ACCEPTANCE.json')['status'].startswith('PASS_SINGLE_CONCENTRATED_STAGE1'),'stage one not accepted')
    # Reuse the original denominator/feature-description formulas, now with the
    # accepted source-row-transported recurrence inputs rather than old IDs.
    module=load(LEGACY/'analysis/describe_revised_inputs.py','r3_data_description')
    args=sys.argv
    sys.argv=['describe','--bundle',str(CAND),'--output',str(DESC),'--recurrence-inputs',str(DIAG)]
    try:module.main()
    finally:sys.argv=args
    desc=read(DESC/'DATA_DESCRIPTION.json');meta=read(DIAG/'METADATA_CONTEXT_READOUT.json')
    desc['inherited_evidence_filter']={
      'revised_entity_source_comparisons.csv':{'old_comparisons':121,'retained_comparisons':121,'removed_comparisons':0,'reason':'Current source-row identity transport; retained condition metadata unchanged'},
      'repeated_pair_source_comparisons.csv':{'old_comparisons':556,'retained_comparisons':558,'removed_comparisons':0,'new_comparisons':2,'reason':'Current repaired graph and source-row alias transport; two comparisons added'}}
    desc['source_graph_repair']={'identities':7,'records':12,'old_edges':5788,'removed_edges':3,'added_edges':19,'current_edges':5804}
    desc['metadata_context_readout']=meta
    dump(DESC/'DATA_DESCRIPTION.json',desc)
    rec=pd.read_csv(CAND/'input_build/data/records_revised.csv');source=pd.read_csv(CAND/'input_build/data/record_sources.csv').set_index('record_id').c2_source_id
    outer=pd.read_csv(CAND/'input_build/data/outer_identity_folds.csv');inner=pd.read_csv(CAND/'input_build/config/inner_identity_folds.csv');sinner=pd.read_csv(CAND/'input_build/config/source_inner_identity_folds.csv')
    vocab=read(CAND/'input_build/config/vocab_manifest.json');selection=read(OLD/'SOURCE_COMPLETION_SELECTION.json')
    support=pd.read_csv(CAND/'input_build/R5_LAYER_SUPPORT.csv');rows=[]
    for x in support.itertuples():
        if x.kind=='internal':
            r,f,i=map(int,re.fullmatch(r'R(\d+)_O(\d+)_I(-?\d+)',x.layer).groups())
            if i==-1:
                fmap=outer.loc[outer.repeat.eq(r)].set_index('identity_hash').outer_fold;mask=rec.identity_hash.map(fmap).ne(f)
            else:
                fmap=inner.loc[inner.repeat.eq(r)&inner.outer_fold.eq(f)].set_index('identity_hash').inner_fold;v=rec.identity_hash.map(fmap);mask=v.notna()&v.ne(i)
            width=vocab[x.layer]['vocabulary_size']
        else:
            f,i=map(int,re.fullmatch(r'S(\d+)_I(-?\d+)',x.layer).groups());r=-1;width=0
            sid=next(z['source_id'] for z in selection['source_holdouts'] if z['source_index']==f)
            if i==-1:
                mask=rec.record_id.map(source).ne(sid)
            else:
                fmap=sinner.loc[sinner.source_index.eq(f)].set_index('identity_hash').inner_fold;v=rec.identity_hash.map(fmap);mask=rec.record_id.map(source).ne(sid)&v.notna()&v.ne(i)
        require(int(mask.sum())==x.training_records,'current saved support count mismatch '+x.layer)
        rows.append({'scope':x.kind,'layer':x.layer,'repeat':r,'outer_fold':f,'inner_fold':i,'training_records':x.training_records,'training_identities':int(rec.loc[mask,'identity_hash'].nunique()),'e1_training_records':x.e1_retained_records,'e0matched_training_records':x.e1_retained_records,'train_pairs':x.training_pairs,'validation_pairs':x.predict_pairs,'vocabulary_size':width})
    pd.DataFrame(rows).to_csv(WORK/'current_inputs/layer_support_census.csv',index=False,lineterminator='\n')
    mask_index=read(CAND/'input_build/MASK_INDEX.json');ordered=rec.sort_values('record_id',kind='mergesort').reset_index(drop=True);members=[]
    for layer,entry in mask_index.items():
        with np.load(CAND/entry['path'],allow_pickle=False) as z:
            training=set(z['training_record_ids'].tolist());matched=set(ordered.iloc[z['e1_training_record_positions']].record_id)
        for rid in ordered.record_id:
            members.append({'layer':layer,'record_id':rid,'eligible_train':rid in training,'e1_support':rid in matched,'e0matched_support':rid in matched})
    pd.DataFrame(members).to_csv(WORK/'current_inputs/layer_record_support.csv',index=False,lineterminator='\n')
    SUMMARY.mkdir(parents=True,exist_ok=True)
    numerical=read(NUM/'NUMERICS_BOOTSTRAP_SUMMARY.json')
    summary={**numerical,'run_id':read(OLD/'EXECUTION_CONTRACT.json')['run_id'],'actual_prediction_components':38,'unique_test_pairs':2825,'test_pair_observations':3450,'internal_models':420,'source_models':33,'new_external_admitted_sources':0,'new_external_evaluation_performed':False,'logical_estimator_fit_coverage':17925,'new_fit_calls':17425}
    dump(SUMMARY/'SUMMARY.json',summary)
    shutil.copyfile(NUM/'primary_estimates.csv',SUMMARY/'primary_estimates.csv')
    return desc

def build_existing_si_tables():
    source=(LEGACY/'tables_source_build/build_si_tables.py').read_text('utf-8')
    begin=source.index('    spec = builder.json(bundle / "spec.json")\n')
    end=source.index('    jobs = spec["jobs"]\n',begin)
    replacement='''    spec = builder.json(bundle / "spec.json")
    require(spec["schema_id"] == "R5_PORTABLE_TRAINING_V1", "unexpected model specification")
    acceptance = builder.json(STAGE_ONE / "FINAL_ACCEPTANCE.json")
    require(acceptance["status"].startswith("PASS_SINGLE_CONCENTRATED_STAGE1"), "current results not accepted")
    contract = builder.json(CURRENT_R5 / "EXECUTION_CONTRACT.json")
    spec["run_id"] = contract["run_id"]
    summary = builder.json(sp / "SUMMARY.json")
    diag = builder.json(dp / "DIAGNOSTICS_RECEIPT.json")
    require(summary["run_id"] == diag["run_id"] == spec["run_id"], "accepted analysis run mismatch")
    require(summary["logical_estimator_fit_coverage"] == 17925 and summary["new_fit_calls"] == 17425, "fit accounting drift")
    require(summary["new_external_admitted_sources"] == 0 and not summary["new_external_evaluation_performed"], "unplanned external result")
    cfg = builder.json(local(bundle, spec["scientific_config"]))
    prep = builder.json(local(bundle, spec["input_preparation_receipt"]))
    desc = builder.json(ddp / "DATA_DESCRIPTION.json")
    records = builder.read(local(bundle, spec["inputs"]["records"]), ["record_id", "identity_hash", "normalized_poi_id", "pair_context_hash"])
    external = builder.read(local(bundle, spec["external"]["records"]))
    source = builder.json(local(bundle, spec["source_holdout"]["selected_sources"]))["source_holdouts"]
    require([r["source_index"] for r in source] == [0, 1, 3], "source indices changed")
    require(not spec["source_holdout"]["new_count_holdout"] and not spec["source_holdout"]["matched_arm_in_source_holdout"], "source scope expanded")
    census_path = bundle / "input_build/config/layer_support_census.csv"
    census = builder.read(census_path, ["scope", "layer", "repeat", "outer_fold", "inner_fold", "training_records", "training_identities", "e1_training_records", "e0matched_training_records", "train_pairs", "validation_pairs", "vocabulary_size"])
    require(all(r["e1_training_records"] == r["e0matched_training_records"] for r in census), "matched support differs")
    require(len(records) == desc["records"] == diag["main_records"] == 699, "record population differs")
    require(desc["pairs"] == diag["main_pairs"] == 5804, "repaired graph population differs")
    require(len(external) == 7, "fixed BRD4 case changed")
'''
    source=source[:begin]+replacement+source[end:]
    source=replace_once(source,'summary["internal_models"] == diag["frozen_final_slots"]','summary["internal_models"] == 420')
    source=replace_once(source,'require(len(excluded) == prep["summary"]["removed_records"] and len({r["record_id"] for r in excluded}) == len(excluded), "exclusion scope differs")','require(len(excluded) == 24 and len({r["record_id"] for r in excluded}) == 24, "historical exclusion scope differs")')
    source=replace_once(source,'["Edges removed from previous continuous graph", str(len(builder.read(bundle / "input_build/source/pairs_5979.csv")) - desc["pairs"]), "Deletion-only update; no new edge is added"],','["Historical 24-record exclusion", "191 edges removed; 5979 to 5788", "Deletion-only step before the present structure repair"],\n        ["Source-depicted graph repairs", "7 identities / 12 records", "One-carbon bridge corrections; Example 17 also adds the source-drawn N-methyl; labels unchanged"],\n        ["Graph update after structure repair", "3 removed / 19 added; net +16", "5788 to 5804 compatible edges under unchanged eligibility rules"],')
    source=replace_once(source,'require(proof["removed_comparisons"] == 0 and proof["old_comparisons"] == proof["retained_comparisons"] == rec["n"] == len(actual), "recurrence reuse proof no longer applies")','require(proof["retained_comparisons"] == rec["n"] == len(actual), "current recurrence support differs")')
    source=replace_once(source,'The deletion-only filter leaves these comparisons unchanged, as documented in DATA_DESCRIPTION.json; the historical full census is not recomputed.','The repaired source-row mapping preserves the 121 record comparisons and adds two pair comparisons, giving 558; the current condition/source relations are recomputed on the corrected graph and documented in DATA_DESCRIPTION.json.')
    source=source.replace('"S18": "reuse after deletion-only retained-comparison proof in DATA_DESCRIPTION.json; no old full c3 recomputation"','"S18": "accepted corrected graph/source-row transport, current condition/source recurrence"')
    adapted=WORK/'build_si_tables_r3.py';write(adapted,source)
    module=load(adapted,'current_si_formatter')
    module.STAGE_ONE=POST/'stage1_acceptance';module.CURRENT_R5=OLD
    original_track=module.Builder.track
    def current_track(self,path):
        p=Path(path).resolve()
        if p.parent==SUMMARY.resolve() and not p.exists():p=SAVED/p.name
        if p==CAND/'input_build/config/layer_support_census.csv':p=WORK/'current_inputs/layer_support_census.csv'
        if p==CAND/'input_build/config/layer_record_support.csv':p=WORK/'current_inputs/layer_record_support.csv'
        if p==CAND/'input_build/config/selected_sources.json':p=OLD/'SOURCE_COMPLETION_SELECTION.json'
        if p.is_relative_to(CAND/'input_build/source') and not p.exists():p=LEGACY/'repro/portable_training'/p.relative_to(CAND)
        return original_track(self,p)
    module.Builder.track=current_track
    result=module.build(SimpleNamespace(bundle=CAND,summary=SUMMARY,diagnostics=DIAG,data_description=DESC,prior_si=PRIOR,output=WORK/'generated',all_results_ready=True))
    print(json.dumps(result),flush=True)
    return module

def extra_tables(prior):
    result={23:section(prior,23)}
    metric=pd.read_csv(NUM/'ripk1_metrics_verified.csv');contrast=pd.read_csv(NUM/'ripk1_contrasts_verified.csv')
    panels=[]
    for layer,label in [('primary42','Primary: 42 prespecified similar pairs'),('secondary45','Secondary: all 45 pairs')]:
        rows=[]
        for rep in R:
            for learner in L:
                for arm in A:
                    if arm=='zero' or (learner=='ridge' and arm=='context_pair_C1'):continue
                    sub=metric.loc[metric.layer.eq(layer)&metric.representation.eq(rep)&metric.learner.eq(learner)&metric.strategy.eq(arm)].set_index('output_scale')
                    rows.append([R[rep],L[learner],A[arm]]+[fmt(sub.loc['raw',field],3)+' / '+fmt(sub.loc['calibrated',field],3) for field in ['MAE','RMSE','strict_DA','half_zero_DA']])
        z=metric.loc[metric.layer.eq(layer)&metric.representation.eq('reference')].iloc[0]
        rows.append(['—','—','Zero']+[fmt(z[field],3)+' / '+fmt(z[field],3) for field in ['MAE','RMSE','strict_DA','half_zero_DA']])
        panels.append('**'+label+'.**\n\n'+md(['Repr.','Learner','Arm','MAE raw/cal','RMSE raw/cal','Strict DA raw/cal','Half-zero DA raw/cal'],rows))
    result[24]='### Table S24. Complete fixed-model RIPK1 case results.\n\n'+'\n\n'.join(panels)+'\n\nRaw and calibrated predictions are averaged separately over all 15 members; each member applies its own inner-OOF scale before averaging. Errors use pDC50 units. Strict and half-zero DA follow the frozen sign/zero rules. The secondary panel contains the same ten molecules and all primary edges. Full-precision metrics, edge/member predictions and model-minus-Zero differences are retained in the current companions.\n'
    rows=[]
    for layer,label in [('primary42','42 primary'),('secondary45','45 secondary')]:
        for rep in R:
            for learner in L:
                cells=[]
                for kind in ['E0matched-E1','Direct-E1']:
                    sub=contrast.loc[contrast.layer.eq(layer)&contrast.representation.eq(rep)&contrast.learner.eq(learner)&contrast.contrast.eq(kind)].set_index('output_scale').MAE_difference
                    cells.append(fmt(sub['raw'],3,True)+' / '+fmt(sub['calibrated'],3,True))
                rows.append([label,R[rep],L[learner],*cells])
    result[25]=table(25,'Prespecified within-case MAE differences.','Positive differences favor E1. E0-matched − E1 is the primary case contrast; Direct − E1 and all model − Zero differences are supplementary. S25_contrasts.csv retains all combinations and scales. Values describe one source/block; no edge-level resampling or multiple-source inference is added.',['Layer','Repr.','Learner','E0-matched − E1 raw/cal','Direct − E1 raw/cal'],rows)
    d=pd.read_csv(MAP/'R6_S26_support.csv');rows=[]
    for x in d.itertuples():rows.append([x.component,int(x.pairs),int(x.pair_endpoint_identities),f'{x.matched_train_records}/{x.matched_train_identities}',fmt(x.primary_support_median_identity_nn,4),fmt(x.zero_mae,4),int(x.n_normalized_sources)])
    result[26]=table(26,'Training support and response scale for all 43 held-out components.','S is the median test-identity nearest-neighbor binary Tanimoto similarity to outer-training E0-matched/E1 support. Matched support excludes singleton blocks; count vocabulary uses all eligible training identities. Component IDs are fixed, and source counts do not establish independent family membership.',['Component','n_pairs','n_pair_endpoint_IDs','Matched-training records/IDs','S','Zero MAE','n_normalized_sources'],rows)
    d=pd.read_csv(MAP/'R6_S27_contrasts.csv');rows=[[x.Component,fmt(x.d_binary,4),fmt(x.d_count,4),fmt(x.u_binary,4),fmt(x.u_count,4)] for x in d.itertuples()]
    result[27]=table(27,'Complete component contrasts for binary and count representations.','$d$ is the mean of three fixed learner-specific MAE(E1) − MAE(E0-matched) contrasts. For $u=(A−B)/(A+B)$, A and B are the three learner MAE means for E1 and E0-matched. We define u=0 for a zero denominator; otherwise we use the ratio without an epsilon. Negative d/u favor E1. These average errors rather than predictions.',['Component','d_binary','d_count','u_binary','u_count'],rows)
    d=pd.read_csv(MAP/'R6_S28_summary.csv');rows=[]
    for x in d.itertuples():rows.append([x.representation,x.learner,fmt(x.theta,4),int(x.E1_better),int(x.ties),int(x.E0matched_better),fmt(x.rho_S_d,4),fmt(x.rho_S_u,4)])
    caption=section(prior,28).split('\n\n')[-1]
    result[28]=table(28,'Finite-population contrast summaries across all 43 components.',caption,['Representation','Learner','$\\theta_{\\mathrm{LOCO}}$','E1 better','Tie','E0-matched better','$\\rho(S,d)$','$\\rho(S,u)$'],rows)
    d=pd.read_csv(MAP/'R6_S29_sensitivity.csv');rows=[]
    labels={'all_43_components':'All 43 components','at_least_5_pair_endpoint_identities':'At least 5 pair-endpoint IDs','delete_one_component_fixed_predictions':'Delete one: 43 fixed-prediction omissions','omit_two_largest_pair_groups':'Omit DEP_006 and DEP_002'}
    for x in d.itertuples():
        span=lambda a,b:fmt(a,4) if a==b else fmt(a,4)+'–'+fmt(b,4)
        rows.append([x.representation,labels[x.population],f'{x.components} each' if x.population=='delete_one_component_fixed_predictions' else int(x.components),span(x.theta_min,x.theta_max),span(x.rho_min,x.rho_max) if x.undefined_rhos==0 else f'undefined: {int(x.undefined_rhos)}'])
    result[29]=table(29,'Prespecified fixed-prediction sensitivity summaries.',section(prior,29).split('\n\n')[-1],['Representation','Population','Components','$\\theta_{\\mathrm{LOCO}}$','$\\rho(S,d)$'],rows)
    repairs=pd.read_csv(MAP/'S30_repair_candidates.csv');rows=[]
    for x in repairs.itertuples():
        rows.append([x.example,x.TPD_ID,int(x.affected_records),x.change,x.bridgehead_configuration,f'A1 p.{int(x.A1_PDF_page)} / A9 p.{int(x.A9_PDF_page)}'])
    result[30]=table(30,'Source-depicted structure repairs in seven training identities.','Twelve source records retain their measurements and recorded conditions. All seven repairs restore the source-drawn one-carbon bridge; Example 17 also restores the benzimidazolone N-methyl. The bridgeheads retain the source-supported R,R configurations; source-unspecified stereochemistry remains unassigned. A1/A9 page numbers identify the inspected local patent PDFs, without asserting the formal scope of an A9 correction. Full old/new SMILES, source rows, corrected record IDs, identity hashes and evidence pointers are in S30_repair_candidates.json. This event is distinct from the earlier seven RIPK1 transcription repairs described in Text S8.',['Example','TPD_ID','Records','Constitutional repair','Bridgeheads','Depiction evidence'],rows)
    intervals=pd.read_csv(NUM/'primary_estimates.csv').set_index('statistic');n=read(NUM/'KEY_MANUSCRIPT_NUMBERS.json');r6=read(MAP/'R6_reading_summary.json')
    rows=[['Compatible graph edges','5788','5804','3 removed and 19 added after structure repair; same eligibility rule'],['Internal test observations / unique pairs','3438 / 2817','3450 / 2825','Updated fixed-member evaluation population'],['Binary E1 lower MAE than E0-matched','9/9','7/9','Learner–repeat direction count'],['Count E1 lower MAE than E0-matched','6/9','6/9','Direction count unchanged'],['Count lower MAE than binary','7/14','6/14','Learner–strategy mean-MAE count']]
    old_intervals={'theta_match_binary':(0.023830,0.001677,0.041662),'theta_match_count':(0.009449,-0.009229,0.029609),'theta_binary':(0.041407,0.023011,0.058667),'theta_count':(0.027093,0.003893,0.054573)}
    for key,label in [('theta_match_binary','Internal θmatch, binary'),('theta_match_count','Internal θmatch, count'),('theta_binary','Internal Direct/E1 θ, binary'),('theta_count','Internal Direct/E1 θ, count')]:
        o=old_intervals[key];x=intervals.loc[key]
        rows.append([label,f'{fmt(o[0])} [{fmt(o[1])}, {fmt(o[2])}]',f'{fmt(x.estimate)} [{fmt(x.lower_95)}, {fmt(x.upper_95)}]','Both matched-support intervals include zero' if 'match' in key else 'Direct/E1 interval remains above zero'])
    for rep,old_theta,old_rho in [('binary',0.0148026646,-0.0921241),('count',0.0089002881,-0.2190591)]:
        x=next(x for x in r6['primary'] if x['representation']==rep)
        rows.append([f'LOCO E1−E0-matched / ρ(S,d), {rep}',f'{fmt(old_theta)} / {fmt(old_rho)}',f"{fmt(float(x['theta_E1_minus_E0matched']))} / {fmt(float(x['spearman_support_delta']))}",'Equal-component mean remains positive; association is descriptive'])
    rows.extend([['Incomplete nonsingleton block graphs','19/74','17/74','Recomputed on the repaired graph'],['Repeated entity–POI pair comparisons','556','558','522 different-condition and 36 incomplete-condition comparisons'],['RIPK1 primary Direct lower MAE than E1','5/6','3/6','All three binary cells; no count cell'],['RIPK1 primary E0-matched lower MAE than E1','6/6','6/6','Single-source case direction retained']])
    result[31]=table(31,'Key results before and after source-graph repair.','Old values identify the preceding manuscript snapshot; new values use the same protocols on the source-supported repaired inputs and their corresponding saved predictions. Intervals are the 2000-draw fixed-prediction component bootstrap; LOCO values are finite-population descriptions. This is a correction/recomputation comparison, not an independent validation experiment. The full mapped table values and evidence paths are retained in S31_old_new_source_map.csv.',['Quantity','Previous snapshot','Repaired inputs','Interpretation'],rows)
    return result

def main_tables():
    # Reuse the established Main Table 1/2 formatter exactly, without its old
    # root-level manuscript collector or historical Table 3 numbering.
    src=load(LEGACY/'source_build/summarize_for_manuscript.py','main_table_format')
    t1=pd.read_csv(NUM/'table1_mae.csv');rows=[];rawmaps={}
    for learner in L:
        for arm in A:
            if arm=='zero' or (learner=='ridge' and arm=='context_pair_C1'):continue
            x=t1.loc[t1.learner.eq(learner)&t1.pipeline.eq(arm)].iloc[0]
            rows.append([L[learner],A[arm],src.fmt(x.mae_binary),src.fmt(x.mae_count),src.fmt(x.absolute_mae_difference,signed=True)])
            rawmaps[('Table1',L[learner],A[arm])]={'Binary MAE':x.mae_binary,'Count MAE':x.mae_count,'Count − binary':x.absolute_mae_difference}
    z=read(NUM/'KEY_MANUSCRIPT_NUMBERS.json')['zero_mean_mae'];rows.append(['Zero','Constant difference',src.fmt(z),src.fmt(z),'0']);rawmaps[('Table1','Zero','Constant difference')]={'Binary MAE':z,'Count MAE':z,'Count − binary':0.0}
    tables={1:src.table(['Learner','Strategy','Binary MAE','Count MAE','Count − binary'],rows)}
    d=pd.read_csv(NUM/'primary_estimates.csv').set_index('statistic');rows=[]
    labels=[('gamma_'+arm,'γ, '+A[arm]+': log(count/binary MAE)') for arm in ['direct_pair','endpoint_E0','endpoint_E0_matched','endpoint_E1','context_pair_C1']]
    labels += [('theta_binary','θ, binary: log(Direct/E1 MAE)'),('theta_count','θ, count: log(Direct/E1 MAE)'),('theta_match_binary','θ match, binary: log(E0-matched/E1 MAE)'),('theta_match_count','θ match, count: log(E0-matched/E1 MAE)'),('theta_support_binary','θ support, binary: log(E0/E0-matched MAE)'),('theta_support_count','θ support, count: log(E0/E0-matched MAE)'),('delta','δ: θ count − θ binary')]
    for key,label in labels:
        x=d.loc[key];rows.append([label,src.fmt(x.estimate),f'[{src.fmt(x.lower_95)}, {src.fmt(x.upper_95)}]']);rawmaps[('Table2',label)]={'Estimate':x.estimate,'95% conditional interval':f'[{x.lower_95:.17g}, {x.upper_95:.17g}]'}
    tables[2]=src.table(['Contrast','Estimate','95% conditional interval'],rows)
    d=pd.read_csv(MAP/'R6_Table3.csv');rows=[]
    for _,x in d.iterrows():
        vals=[x['Direct_MAE'],x['E0_MAE'],x['E0-matched_MAE'],x['E1_MAE'],x['Zero_MAE']];rows.append([x.Representation,x.Learner]+[fmt(v,4) for v in vals]);rawmaps[('Table3',x.Representation,x.Learner)]=dict(zip(['Direct','E0','E0-matched','E1','Zero'],vals))
    tables[3]=src.table(['Representation','Learner','Direct','E0','E0-matched','E1','Zero'],rows)
    d=pd.read_csv(NUM/'table4_ripk1_primary42.csv');rows=[]
    for rep in R:
        for learner in L:
            x=d.loc[d.representation.eq(rep)&d.learner.eq(learner)].iloc[0];vals=[x.endpoint_E0_matched,x.endpoint_E1,x.direct_pair,x.Zero_MAE,x.E0matched_minus_E1];rows.append([R[rep],L[learner]]+[fmt(v,3) for v in vals]);rawmaps[('Table4',R[rep],L[learner])]=dict(zip(['E0-matched','E1','Direct','Zero','E0-matched − E1'],vals))
    tables[4]=src.table(['Repr.','Learner','E0-matched','E1','Direct','Zero','E0-matched − E1'],rows)
    for i,t in tables.items():write(WORK/f'Table{i}.md',t+'\n')
    write(WORK/'MAIN_TABLES.md','\n\n'.join(f'## Table {i}\n\n{t}' for i,t in tables.items())+'\n')
    return rawmaps

def copy_companions():
    index=[]
    def add(sid,source,name=None,scope='',pointer=False):
        source=Path(source);target=name or f'S{sid:02d}_{source.name}'
        if pointer:
            index.append({'table':f'S{sid}','file':'','source':source.as_posix(),'scope':scope,'storage':'local relative pointer','relative_pointer':Path(__import__('os').path.relpath(source,COMP)).as_posix()})
        else:
            shutil.copyfile(source,COMP/target);index.append({'table':f'S{sid}','file':target,'source':source.as_posix(),'scope':scope,'storage':'copied current artifact','relative_pointer':''})
    oldidx=pd.read_csv(WORK/'generated/COMPANION_INDEX.csv').fillna('')
    for x in oldidx.itertuples():add(int(x.table[1:]),WORK/'generated'/x.file,Path(x.file).name,x.scope)
    for sid,names in [(23,['authority/case_records.csv']), (24,['member_predictions.csv','ensemble_predictions.csv','ensemble_metrics.csv']),(25,['contrasts.csv'])]:
        for name in names:
            p=OLD/'ripk1_case'/name
            if p.exists():add(sid,p,'S%02d_%s'%(sid,p.name),'Current RIPK1 frozen records or corrected internal-member case outputs')
    # The fixed case authority source can retain its original filename.
    p=OLD/'ripk1_case/FROZEN_MANIFEST.json';add(23,p,'S23_FROZEN_MANIFEST.json','Current case/member freeze and authority pointers')
    for path in (LEGACY/'admission_followup/schema_compatibility/inputs').glob('*'):
        if path.is_file():add(23,path,'S23_'+path.name,'Unchanged frozen RIPK1 case inputs; current freeze binds these exact inputs')
    for name in ['applicability_nearest699.csv','count_unknown_environments.csv']:
        add(23,OLD/'ripk1_case'/name,'S23_'+name,'Current corrected-training applicability and vocabulary readout')
    add(23,OLD/'ripk1_case/authority/AUTHORITY_BINDINGS.json','S23_AUTHORITY_BINDINGS.json','Current retained case authority and source binding')
    r6=BASE/'r6_budget_b_20261012/summary'
    for path in (r6/'results').glob('*.csv'):
        d=pd.read_csv(path)
        target=COMP/f'S29_{path.name}'
        if 'scenario' in d:
            d=d.loc[d.scenario.eq('corrected_shadow')];d.to_csv(target,index=False,lineterminator='\n');scope='scenario=corrected_shadow only; accepted R6 results';storage='filtered current CSV'
        else:
            shutil.copyfile(path,target);scope='Accepted R6 current table; historical comparison columns explicitly labeled where present';storage='copied current artifact'
        index.append({'table':'S26-S29','file':target.name,'source':path.as_posix(),'scope':scope,'storage':storage,'relative_pointer':''})
    for path in (r6/'corrected_metadata').glob('*.csv'):add(26,path,'S26_'+path.name,'Accepted current R6 component/input metadata')
    for path in (r6/'corrected_metadata').glob('*.json'):add(26,path,'S26_'+path.name,'Accepted current R6 metadata provenance')
    for path in [BASE/'r6_budget_b_20261012/ROOT_STAGE_ACCEPTANCE.json',r6/'SUMMARY.json']:add(29,path,'S29_'+path.name,'Accepted R6 execution/result entry')
    for name in ['R6_Table3.csv','R6_S26_support.csv','R6_S27_contrasts.csv','R6_S28_summary.csv','R6_S29_sensitivity.csv']:add(int(name[4:6]) if name.startswith('R6_S') else 29,MAP/name,name,'Accepted corrected_shadow reader table')
    for name in ['S30_repair_candidates.csv','S30_repair_candidates.json']:add(30,MAP/name,name,'Accepted seven-identity structure repair; includes current corrected record IDs')
    add(31,NUM/'primary_estimates.csv','S31_current_primary_estimates.csv','Current 14 contrasts and conditional intervals')
    add(31,DIAG/'METADATA_CONTEXT_READOUT.json','S31_METADATA_CONTEXT_READOUT.json','Repaired graph and recurrence comparison')
    add(31,POST/'stage1_acceptance/FINAL_ACCEPTANCE.json','S31_STAGE1_ACCEPTANCE.json','Single scientific numeric/diagnostic acceptance')
    return index

def fill_old_new_map(rawmaps):
    data=pd.read_csv(MAP/'S31_old_new_source_map.csv').fillna('');n=read(NUM/'KEY_MANUSCRIPT_NUMBERS.json');r6=read(MAP/'R6_reading_summary.json')
    vals={'compatible_pairs':5804,'internal_test_observations':3450,'internal_unique_test_pairs':2825,'binary_E1_better_than_E0matched_repeat_cells':'7/9','count_E1_better_than_E0matched_repeat_cells':'6/9','count_lower_MAE_than_binary':'6/14','internal_persistent_zero_observations':19,'internal_restored_observations':71,'internal_both_nonzero_observations':3360,'all_record_binary_input_classes':439,'all_record_count_input_classes':493,'all_edge_binary_zero_differences':135,'all_edge_count_zero_differences':26,'bootstrap_contributing_components':38,'incomplete_nonsingleton_blocks':'17/74','recurring_identity_POI_record_comparisons':121,'recurring_entity_POI_pair_comparisons':558,'recurring_pair_different_observed_condition':522,'all60layer_vocab_range':'2196-2735','outer15layer_vocab_range':'2592-2735','BRD4_count_lower_MAE':'2/14','BRD4_count_higher_strict_direction':'13/14','BRD4_count_higher_halfcredit_direction':'7/14'}
    for idx,row in data.iterrows():
        key=row.comparison
        if key in vals:value=vals[key]
        elif key.startswith('Table'):
            parts=key.split('|');rk=tuple(parts[:-1]);require(rk in rawmaps and parts[-1] in rawmaps[rk],'new table-map key absent '+key);value=rawmaps[rk][parts[-1]]
        elif key.startswith('LOCO|'):
            _,rep,field=key.split('|');value=next(x for x in r6['primary'] if x['representation']==rep)[field]
        else:raise RuntimeError('Unmapped old/new key '+key)
        data.loc[idx,'new_value']=str(value);data.loc[idx,'new_status']='ACCEPTED_CURRENT_INPUT';data.loc[idx,'new_source']=('current accepted R3 tables and stage-one CSVs' if key.startswith('Table') else row.new_source)
    data.to_csv(COMP/'S31_old_new_source_map.csv',index=False,lineterminator='\n')

def update_si_text(text,desc):
    """Replace factual paragraphs only; equations and method definitions persist."""
    changes=[]
    def para(prefix,new):
        nonlocal text
        pattern=re.compile(r'^'+re.escape(prefix)+r'[^\n]*(?:\n(?!\n)[^\n]+)*',re.M)
        matches=list(pattern.finditer(text));require(len(matches)==1,'paragraph identity mismatch '+prefix)
        old=matches[0].group();text=text[:matches[0].start()]+new+text[matches[0].end():];changes.append({'prefix':prefix,'old':old,'new':new})
    def literal(old,new):
        nonlocal text
        text=replace_once(text,old,new);changes.append({'old':old,'new':new})
    para('Pairs join distinct chemical identities',
      'Pairs join distinct chemical identities within a refined recorded-condition block when their binary Morgan Tanimoto similarity is at least 0.5. Known different exposure times and corrected different sources are separated. The preceding condition revision removed 36 of the historical 6015 edges: 18 adjoining excluded coarse-category records, 17 joining explicitly different exposure times, and one joining corrected different sources. The 24-record exclusion then removed 191 edges from the 5979-edge graph, producing the historical 5788-edge snapshot without adding pairs. The present source-depicted structure repair removes three of those edges and adds nineteen under the same eligibility rule, yielding 5804 compatible edges. The development pool retains 699 records, 511 identities and 172 blocks; 598 records are graph endpoints and the other 101 may contribute to endpoint training. All compatible edges define the primary analysis. The threshold |y| ≥ 1 remains a supplementary stratum, with 825 current cliff edges.')
    anchor='### Supplementary Text S2.'
    repair='Seven training identities, represented by twelve source records from WO2024183650A1/A9, were corrected by comparison with the source drawings. Each repair restores an omitted one-carbon internal bridge; Example 17 additionally restores the source-drawn benzimidazolone N-methyl. Source-supported bridgehead configurations are R,R, while stereochemistry not specified by the source remains unassigned. Measurements and recorded-condition fields are unchanged. Table S30 and its companions bind the old/new structures, source-row identifiers and corrected record identities; Table S31 compares the corresponding old and repaired-input results. These are the training-source repairs, distinct from the earlier RIPK1 transcription corrections in Text S8.\n\n'
    text=replace_once(text,anchor,repair+anchor)
    literal('19 of the 74 full-data blocks containing at least two records are incomplete','17 of the 74 full-data blocks containing at least two records are incomplete')
    para('The outer evaluation uses three repetitions',
      'The outer evaluation uses three repetitions of five folds. All records sharing a complete-graph chemical identity are assigned to the same fold across POIs and contexts. The retained outer maps use seeds 20261007, 20261008, and 20261009. Source-row correspondence transports the fixed member assignments to the corrected identities without a new fold allocation; original outer-cell inner memberships are retained. In every training, validation, or test layer, a pair is used only if both endpoint identities belong to that layer. Edges crossing folds are omitted from that evaluation. The three repetitions therefore test 2825 different edges through 3450 observations, rather than every one of the 5804 compatible edges (Table S2).')
    para('The post hoc diagnostic assigns observations',
      'The post hoc diagnostic assigns observations by their actual feature differences under the binary input and relevant count training vocabulary. “Persistent zero” has $d_{\\mathrm{binary}}=d_{\\mathrm{count}}=0$; “restored” has $d_{\\mathrm{binary}}=0$ and $d_{\\mathrm{count}}\\ne0$; “both nonzero” has both differences nonzero. Membership uses exact input equality without labels, errors or predicted zeros. The 3450 internal test observations comprise 19 persistent-zero, 71 restored and 3360 both-nonzero observations, covering 17, 59 and 2749 different edges, respectively (Table S13). The 511 retained identities form 439 binary and 493 raw count input classes, with 41 and 13 mixed-identity classes. Across all 5804 compatible edges, 135 have a zero binary difference and 26 have a zero raw count difference. These full-graph counts differ from actual test-observation counts under each training vocabulary. Identities, blocks and dependency components may overlap across groups, so their support counts are not independent sample sizes.')
    literal('0.015146 pDC50 for binary inputs and 0.003255 pDC50 for actual count inputs','0.015091 pDC50 for binary inputs and 0.003244 pDC50 for actual count inputs')
    loss=pd.read_csv(DIAG/'posthoc_internal_calibrated_loss_contributions.csv')
    x=loss.loc[loss.learner.eq('random_forest')&loss.pipeline.eq('endpoint_E1')].iloc[0]
    para('Count inputs had a lower calibrated restored-group contribution',
      f'Count inputs have a lower calibrated restored-group contribution in 11 of the 14 eligible learner–strategy comparisons, but a lower both-nonzero contribution in only five. Overall MAE is lower in six comparisons and higher in eight (Table S14). For RF E1, the restored contribution is {fmt(x.restored)} pDC50, whereas the both-nonzero contribution is {fmt(x.both_nonzero,6,True)} pDC50, yielding an overall count−binary difference of {fmt(x.overall_count_minus_binary_mae,6,True)} pDC50. Recovering a nonzero structural difference therefore does not ensure lower total error. Table S14 and `S14_posthoc_internal_calibrated_three_group_details.csv` retain every arm, group-specific Zero comparison and repetition range; the input-defined decomposition does not isolate a causal representation effect.')
    loss=pd.read_csv(DIAG/'posthoc_external_calibrated_loss_contributions.csv');xe=loss.loc[loss.learner.eq('xgboost')&loss.pipeline.eq('endpoint_E0')].iloc[0];xm=loss.loc[loss.learner.eq('xgboost')&loss.pipeline.eq('endpoint_E0_matched')].iloc[0]
    para('The seven-compound series contained no persistent-zero edges',
      f'The seven-compound series contains no persistent-zero edges, ten restored edges and eleven both-nonzero edges. The empty persistent-zero group has NA own-group metrics and zero contribution. Among 14 calibrated ensemble comparisons, count inputs lower the restored contribution in seven and the both-nonzero contribution in two; overall MAE is lower in two and higher in twelve (Table S22). The two overall reductions are XGB E0 and E0-matched: restored contributions of {fmt(xe.restored)} and {fmt(xm.restored)} pDC50 outweigh both-nonzero increases of {fmt(xe.both_nonzero,6,True)} and {fmt(xm.both_nonzero,6,True)}, giving total differences of {fmt(xe.overall_count_minus_binary_mae)} and {fmt(xm.overall_count_minus_binary_mae)} pDC50. Every contribution uses all 21 edges as its denominator. Full group errors, own-group Zero comparisons, direction scores and zero counts remain in `S22_posthoc_external_calibrated_three_group_details.csv`. Each ensemble prediction averages 15 calibrated member predictions before loss is computed. Mean member loss is saved separately. The shared endpoints and previously examined single series limit this analysis to descriptive diagnostics.')
    sm=pd.read_csv(NUM/'source_metrics.csv');source=sm.loc[sm.scenario.eq('source_unseen')];cal=source.loc[source.scale.eq('calibrated')];zero=source.loc[source.pipeline.eq('zero')].set_index(['holdout_id','learner']).mae
    c=cal.copy();c['zero']=c.set_index(['holdout_id','learner']).index.map(zero);good=c.loc[c.mae.lt(c.zero)]
    cells='; '.join(f"{hid.replace('C2_','')} {L[learner]} {A[arm]}" for hid,learner,arm in good[['holdout_id','learner','pipeline']].itertuples(index=False,name=None))
    para('The retained source evaluations contained',
      f'The retained source evaluations contain 2222 edges for S0, 253 for S1 and 93 for S3 (Tables S15 and S16). Direct has lower calibrated MAE than E1 in four of nine learner–source comparisons. Against source-specific Zero, Direct is lower in one of nine comparisons, E0 in three of nine, E1 in two of nine and C1 in two of six eligible tree comparisons: {cells}. Thus eight of 33 calibrated source-model cells are below Zero. All remaining cells, including those above Zero, remain in Table S16 and the complete raw/calibrated companion.')
    def s3values(scene,arm):
        values=sm.loc[sm.holdout_id.eq('C2_S3')&sm.scenario.eq(scene)&sm.pipeline.eq(arm)&sm.scale.eq('calibrated' if arm!='zero' else 'raw')].set_index('learner').mae
        return ', '.join(fmt(values[l]) for l in L)
    para("S3's strict 84-edge subset had calibrated E1 MAE",
      f"S3's strict 84-edge subset has calibrated E1 MAE of {s3values('source_identity_unseen','endpoint_E1')} pDC50 for Ridge, RF and XGB, respectively, against Zero at 0.865115. On its separate nine-edge one-identity-seen subset, E1 MAEs are {s3values('one_identity_seen','endpoint_E1')}, against Zero at 0.628353. These subsets use the same fitted models; their different errors cannot be attributed solely to identity visibility. S1 also has a training-unseen TBK1 target. These nonrandom sources, one seed, uneven support and shared endpoints do not establish reliable cross-source deployment or a source-population effect.")
    literal('Repeated entity-pair comparisons number 556; all share normalized source clusters, while 520 differ in cell conditions','Repeated entity-pair comparisons number 558; all share normalized source clusters, while 522 differ in observed conditions')
    literal('although its output classes ranged from nine to 21','although its output classes ranged from eight to 21')
    em=pd.read_csv(NUM/'external_metrics.csv');x=em.loc[em.learner.eq('random_forest')&em.pipeline.eq('direct_pair')&em.scale.eq('calibrated')].set_index('representation')
    para('Removing ensemble zeros did not consistently reduce error',
      f'Removing ensemble zeros does not consistently reduce error. Count has lower calibrated MAE in two of fourteen learner–strategy comparisons, higher strict directional accuracy in thirteen and higher half-credit accuracy in seven (Table S20). For RF Direct, strict accuracy changes from {round(x.loc["binary","DA_strict"]*21)}/21 to {round(x.loc["count","DA_strict"]*21)}/21, while MAE changes from {fmt(x.loc["binary","mae"])} to {fmt(x.loc["count","mae"])} pDC50. Direction recovery and error magnitude therefore give different assessments in this previously examined series.')
    con=pd.read_csv(DIAG/'external_consistency.csv');b=con.loc[con.representation.eq('binary')&con.learner.eq('xgboost')&con.pipeline.eq('context_pair_C1')].iloc[0];c=con.loc[con.representation.eq('count')&con.learner.eq('xgboost')&con.pipeline.eq('direct_pair')].iloc[0]
    para('Projection was a separate label-free operation',
      f'Projection is a separate label-free operation on the saved ensemble graph (Table S10). It preserves the ten binary and zero count numerical-zero predictions in this case, but MAE changes are not uniformly favorable. Binary XGB C1 changes from {fmt(b.calibrated_mae)} to {fmt(b.calibrated_projected_mae)} pDC50 after projection, whereas count XGB Direct changes from {fmt(c.calibrated_mae)} to {fmt(c.calibrated_projected_mae)}. Endpoint arms and Ridge Direct retain only numerical consistency residues. Projected potentials describe relative predictions on the seven-node graph rather than absolute endpoint potency or individual-compound uncertainty.')
    # Source/native data facts are preserved; only scientific execution claims
    # and current artifact routes are updated to the accepted repaired-input run.
    para('Scientific model fitting used Python',
      'Scientific model fitting used Python 3.11.15, NumPy 1.26.4, pandas 3.0.3, scikit-learn 1.9.0 and XGBoost 3.0.1; count feature generation used RDKit 2026.03.3. RF and Ridge used CPU execution; XGB used histogram trees with CUDA. Table S4 retains the fixed threads and learner settings. The R5 protocol has logical coverage of 17,925 fits: 16,500 internal and 1425 source-holdout fits. The repaired-input execution closed 17,425 new fits and retained 500 proven reusable fits, comprising the 25-fit first internal parent and 475 source-S0 fits. The separately recorded historical 58-fit partial execution is not added to this identity. The completed population contains 420 internal final models (28 combinations × 15 splits) and 33 binary source final models (11 combinations for each retained source). Each cell evaluates eight candidates over three inner folds plus a final fit, with five separate refits in each tree-E1 unit; the full source budget is 225 fits for Ridge and 600 for each tree learner. Zero and calibration do not fit predictive models. Both cases use the current 420 internal members without new fits. Diagnostics, projection, conditional resampling and loss accounting also use fixed saved predictions.')
    para('The retained 4184-row snapshot was matched byte-for-byte',
      'The retained 4184-row snapshot was matched byte-for-byte to an accessible TACK snapshot retrieved on 13 September 2026. Its original acquisition revision and authoritative retrieval timestamp were not preserved. Current TACK conversion code explains range-midpoint and condition-assignment behavior without recovering the historical generating commit. The reproduction chain starts from the retained snapshot and applies the inherited curation, explicit 24-record exclusion list, source-supported graph repairs and frozen member correspondence before features and analysis. The earlier input reconstruction was executed from an unrelated working directory and compared with its input snapshot. The present correction is separately bound through source-row correspondence, current record/pair maps and saved prediction receipts, rather than claiming a new recovery of the upstream snapshot history. Source pointers and reconstruction instructions do not grant redistribution rights for upstream data or manually transcribed structures.')
    literal('current 699-record/5788-edge comparison','current 699-record/5804-edge comparison')
    metric=pd.read_csv(NUM/'ripk1_metrics_verified.csv');contrast=pd.read_csv(NUM/'ripk1_contrasts_verified.csv')
    def casediffs(rep,layer='primary42',kind='E0matched-E1',scale='calibrated'):
        x=contrast.loc[contrast.representation.eq(rep)&contrast.layer.eq(layer)&contrast.contrast.eq(kind)&contrast.output_scale.eq(scale)].set_index('learner').MAE_difference
        return ', '.join(fmt(x[l],3,True) for l in L)
    para('The primary E0-matched−E1 contrast was negative',
      f'The primary E0-matched−E1 contrast is negative for every learner in both representations and output scales. For Ridge, RF and XGB, calibrated differences are {casediffs("binary")} pDC50 for binary and {casediffs("count")} for count inputs. All six differences remain negative on the secondary 45-pair layer (Table S25), ranging from −0.177416 to −0.025434 pDC50. After member-specific calibration and averaging, Direct has lower MAE than E1 in the three binary cells and higher MAE in all three count cells on the primary layer. On the secondary layer it is lower in four of six cells, including count Ridge; the other two count cells are higher. Raw and calibrated differences remain separate in Table S25.')
    para('Against Zero, E0 was lower in every representation–learner cell',
      'For calibrated predictions on the primary layer, E0 has lower MAE than Zero in all six representation–learner cells; E0-matched does so in five, with count Ridge as the exception. Among E1 and Direct ensembles, only binary Ridge is below Zero. All four C1 ensembles are above Zero. These baseline direction counts also hold on the secondary layer. Raw outputs and every arm remain in Table S24; no model is selected by case performance. The fitted ensembles have no numerical-zero predictions on either layer, so strict and half-credit scores coincide; Zero scores are 0 and 0.5, respectively (`S24_ensemble_metrics.csv`).')
    para('This supplementary protocol extends the earlier internal',
      'The source-component extension uses the same repaired input version as the internal and three-source analyses, retaining the endpoint labels, exclusions and original scientific definitions. The parent data comprise 699 records, 511 complete identities and 5804 compatible canonical pairs. The 108 reference strings map to 89 normalized attribution labels, including accepted DOI/PMID and patent aliases. Records sharing a normalized attribution label or complete identity are joined transitively into 82 components. These are operational source/shared-identity units; complete publication, patent-family and chemical-series independence remains unresolved.')
    para('All 43 components with at least one fixed pair',
      'All 43 components with at least one fixed pair, at least two pair-endpoint identities and nonempty required inner support enter the evaluation. They contain 632 records and 469 complete identities, including all 5804 pairs. The other 39 components have no test-pair MAE but participate in training and vocabulary construction when allowed by strategy. Small groups are not removed according to labels or performance. DEP_006 contains 2222 pairs and DEP_002 contains 1672; together they account for 67.1% of parent edges. The primary aggregation gives components equal weight.')
    para('The protocol allocates 43 × 2 × 3 × 4',
      'The protocol allocates 43 × 2 × 3 × 4 = 1032 final representation–learner–strategy slots. Ordinary training units use one fit and tree E1 units use five; eight candidates over three inner folds plus outer refitting define the original 43,000-fit logical plan (41,280 candidate and 1720 final-refit calls). The accepted corrected-input execution retains 1008 newly fitted final models and 24 proven reusable final models across those 1032 slots, with 28,560 newly completed fit calls. Exact reuse and execution receipts remain in the reproduction records. Calibration, Zero, support and fixed-prediction summaries add no predictive-model fit. The current prediction table contains 139,296 rows (5804 pairs × 24 representation–learner–strategy combinations), with every pair tested once in its held-out component per combination. The previously accepted completion and collection checks bind these current outputs; no training is repeated for manuscript integration.')
    literal('Its original numbers and labels are retained rather than relabeled as LOCO estimates.','Its case estimates retain their original subtraction order and labels rather than being relabeled as LOCO estimates.')
    r6=read(MAP/'R6_reading_summary.json');s28=pd.read_csv(MAP/'R6_S28_summary.csv');s29=pd.read_csv(MAP/'R6_S29_sensitivity.csv')
    p={x['representation']:x for x in r6['primary']}
    para('All 43 pair-containing components were retained in the analysis',
      f'All 43 pair-containing components remain in the analysis. The equal-component, three-learner mean E1 − E0-matched difference is {fmt(float(p["binary"]["theta_E1_minus_E0matched"]),10,True)} for binary and {fmt(float(p["count"]["theta_E1_minus_E0matched"]),10,True)} for count inputs, in pDC50. Both favor E0-matched and have the opposite favorable sign from the internal log(E0-matched/E1) contrast. At |d| ≤ 10⁻¹² ties, binary has 20 E1-better, 19 E0-matched-better and four tied components; count has 24, 18 and one. All six learner-specific mean differences are positive, while component-level directions remain heterogeneous (Table S28). Learner means average errors rather than predictions.')
    u=s28.loc[s28.learner.eq('Three-learner mean')].set_index('representation').rho_S_u
    para('Spearman(S,d) was −0.0921241',
      f'Spearman(S,d) is {fmt(float(p["binary"]["spearman_support_delta"]),7)} for binary and {fmt(float(p["count"]["spearman_support_delta"]),7)} for count. Full-population Spearman(S,u) is {fmt(u["binary"],7)} and {fmt(u["count"],7)}, respectively. DEP_028 retains all 13 pairs with Zero MAE exactly zero. No Zero-normalized division or epsilon is used: the bounded u denominator is the sum of the two model errors.')
    subset=s29.loc[s29.population.eq('at_least_5_pair_endpoint_identities')].set_index('representation');deletion=s29.loc[s29.population.eq('delete_one_component_fixed_predictions')].set_index('representation')
    para('All prespecified mean-contrast sensitivity summaries retained',
      f'All prespecified mean-contrast sensitivities retain the E0-matched direction (Table S29). For the 25 components with at least five pair-endpoint identities, Spearman(S,d) is {fmt(subset.loc["binary","rho_min"],7)} for binary and {fmt(subset.loc["count","rho_min"],7)} for count. The complete one-component-deletion rho ranges are {fmt(deletion.loc["binary","rho_min"])} to {fmt(deletion.loc["binary","rho_max"])} for binary and {fmt(deletion.loc["count","rho_min"])} to {fmt(deletion.loc["count","rho_max"])} for count. These fixed-prediction influence ranges do not quantify retraining uncertainty or test a population correlation. No subgroup, refit, prediction, nominal interval, p value or strategy-selection model is added after error readout.')
    para('The local companion index, si_companions/COMPANION_INDEX.csv',
      'The local companion index, si_companions/COMPANION_INDEX.csv, binds the current internal, source, case and corrected-component outputs. Component companions retain the complete corrected prediction table, component and learner metrics, macro and pair-pooled summaries, Zero, relative-scale and all prespecified sensitivities, together with census, pair mapping, pre-prediction support and acceptance receipts. Tables S26–S29 summarize those accepted corrected_shadow records. S30 supplies the seven-identity repair mapping and S31 the old/new comparison. The reproduction materials retain the execution and reuse chain; no second fit is required for pair-pooled or sensitivity summaries. This remains a private local review package, with public access and upstream redistribution governed by the Data Availability Statement.')
    pd.DataFrame(changes).fillna('').to_csv(WORK/'SI_NUMERIC_TEXT_CHANGES.csv',index=False,lineterminator='\n')
    return text

def integrate_remaining(desc,formatter):
    prior=PRIOR.read_text('utf-8');tables=(WORK/'generated/TABLES.md').read_text('utf-8')
    extra=extra_tables(prior);tables+='\n\n'+'\n\n'.join(extra[i] for i in range(23,32))+'\n'
    write(WORK/'ALL_SI_TABLES.md',tables)
    rawmaps=main_tables();index=copy_companions();fill_old_new_map(rawmaps)
    index.append({'table':'S31','file':'S31_old_new_source_map.csv','source':(MAP/'S31_old_new_source_map.csv').as_posix(),'scope':'162 old manuscript values matched to current accepted values','storage':'current mapped CSV','relative_pointer':''})
    pd.DataFrame(index).to_csv(COMP/'COMPANION_INDEX.csv',index=False,lineterminator='\n')
    header=prior.split('### Table S1.')[0].replace('Table labels S1–S29','Table labels S1–S31')
    text=prior[prior.index('## Supplementary Texts'):]
    text=update_si_text(text,desc)
    write(R3/'source/Supporting_Information.md',header+tables+'\n'+text.rstrip()+'\n')
    receipt={'status':'R3_SI_TABLES_AND_NUMERIC_TEXT_INTEGRATED_PENDING_FINAL_OWNER_ACCEPTANCE','si_tables':31,'main_table_fragments':4,'companion_entries':len(index),'source':(R3/'source/Supporting_Information.md').as_posix(),'old_new_map_rows':len(pd.read_csv(COMP/'S31_old_new_source_map.csv')),'stage1_acceptance_reused':(POST/'stage1_acceptance/FINAL_ACCEPTANCE.json').as_posix(),'models_loaded':0,'fits':0,'new_predictions':0,'bootstrap_repeated':False,'humanize_done':False,'docx_built':False,'encoding':'UTF-8 without BOM; LF','remaining':'Owner humanize and one final integrated/rendered acceptance; external source/author/release gates remain separate'}
    dump(WORK/'INTEGRATION_RECEIPT.json',receipt);print(json.dumps(receipt),flush=True)

def main():
    WORK.mkdir(parents=True,exist_ok=True);COMP.mkdir(parents=True,exist_ok=True)
    desc=prepare_data();formatter=build_existing_si_tables()
    print('S1-S22 current table blocks complete; proceeding to current cases/R6 and prose numbers.',flush=True)
    integrate_remaining(desc,formatter)

if __name__=='__main__':main()
