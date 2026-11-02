"""Frozen Ridge objective sensitivity using supplied candidate OOF caches."""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '4'
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import inner_objective as helper
from internal_branch import InternalBranch
from loco_branch import LocoBranch
from scoring import require, validate_population, evaluate, winner
from summarize import summarize

OBJECTIVES = ('pair_micro','component_macro')
CID = tuple(f'ridge_{i:02d}' for i in range(8))
TOL = 1e-12

def save_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--score-only',action='store_true',help='Stop after cached OOF rescoring and baseline replay; no fitting or prediction')
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    from portable import check_versions
    check_versions(json.loads((Path(__file__).resolve().parents[1]/'core/scientific_config.json').read_text(encoding='utf-8')))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    def resolve(value):
        path=Path(value)
        return (path if path.is_absolute() else manifest_path.parent/path).resolve()
    out=args.output_dir.resolve()
    require(not out.exists() or not any(out.iterdir()),'Output must be new or empty')
    out.mkdir(parents=True,exist_ok=True)
    if 'internal_manifest' in manifest:
        internal={'manifest':resolve(manifest['internal_manifest'])}
    else:
        internal={key:resolve(value) for key,value in manifest['internal'].items()}
    branches=[InternalBranch(**internal,output_root=out),LocoBranch(resolve(manifest['loco_manifest']),out)]
    slots=[(b,s) for b in branches for s in b.slots()]
    require(len(slots)==232,'Expected 232 outer pipeline slots')
    test_data,domain_ids,pair_authority={},{},{}
    for branch,slot in slots:
        data=branch.load_test(slot)
        ids,groups,y,raw=validate_population(data,slot['slot_id'])
        test_data[slot['slot_id']]=data
        domain_ids.setdefault(slot['domain'],set()).update(ids)
        for pid,group,target in zip(ids,groups,y):
            if pid in pair_authority:
                old_group,old_target=pair_authority[pid]
                require(group==old_group and abs(float(target)-old_target)<=TOL,'Cross-branch identity/component/target conflict')
            pair_authority[pid]=(group,float(target))
    common=sorted(domain_ids['internal'] & domain_ids['loco'])
    require(len(domain_ids['internal'])==2825 and len(domain_ids['loco'])==5804,'Native population drift')
    save_json(out/'TEST_POPULATION_FREEZE.json',{'common_pair_ids':common,'common_count':len(common),
                                               'internal_unique':2825,'loco_unique':5804})
    rows,selections,replay=[],[],[]
    with threadpool_limits(limits=4):
        for branch,slot in slots:
            candidates={obj:[] for obj in OBJECTIVES}
            reference=None
            baseline='pair_micro' if slot['domain']=='internal' else 'component_macro'
            for cid in CID:
                data=branch.load_oof(slot,cid)
                ids,groups,y,raw=validate_population(data,slot['slot_id']+'/'+cid)
                expected_groups=np.array([pair_authority[x][0] for x in ids])
                expected_y=np.array([pair_authority[x][1] for x in ids])
                require(np.array_equal(groups,expected_groups) and np.max(np.abs(y-expected_y))<=TOL,'OOF authority conflict')
                if reference is None:
                    reference=(ids.copy(),groups.copy(),y.copy())
                require(all(np.array_equal(a,b) for a,b in zip(reference,(ids,groups,y))),'Candidate OOF population/label drift')
                require(not (set(ids)&set(test_data[slot['slot_id']]['pair_ids'])),'OOF/test overlap')
                if slot['domain']=='loco':
                    require(len(np.unique(groups))==42,'LOCO OOF components differ')
                for objective in OBJECTIVES:
                    row={key:slot[key] for key in ('slot_id','domain','representation','pipeline','repeat','outer_fold')}
                    row.update(candidate_id=cid,objective=objective,oof_pairs=len(ids),oof_components=len(np.unique(groups)),
                               **evaluate(helper,y,raw,groups,objective))
                    candidates[objective].append(row)
                    rows.append(row)
                observed,saved=candidates[baseline][-1],slot['old_scores'][cid]
                c_gap=abs(observed['scale_c']-saved['scale_c'])
                score_gap=abs(observed['selection_score']-saved['selection_score'])
                replay.append({'slot_id':slot['slot_id'],'candidate_id':cid,'scale_abs_gap':c_gap,'score_abs_gap':score_gap})
                require(c_gap<=TOL and score_gap<=TOL,'Baseline replay failed')
            for objective in OBJECTIVES:
                chosen=dict(winner(candidates[objective]))
                chosen.update(old_candidate_id=slot['old_candidate_id'],old_scale=slot['old_scale'],
                              requires_new_fit=chosen['candidate_id']!=slot['old_candidate_id'])
                if objective==baseline:
                    require(chosen['candidate_id']==slot['old_candidate_id'],'Baseline winner changed')
                selections.append(chosen)
        require(len(rows)==3712 and len(replay)==1856 and len(selections)==464,'Scoring accounting differs')
        required=sum(row['requires_new_fit'] for row in selections)
        require(required<=232,'Frozen fit cap exceeded')
        results=out/'results'
        results.mkdir()
        for name,values in [('CANDIDATE_SCORES',rows),('SELECTIONS',selections),('BASELINE_REPLAY',replay)]:
            pd.DataFrame(values).to_csv(results/(name+'.csv'),index=False,lineterminator='\n')
        save_json(out/'BASELINE_GATE.json',{'status':'PASS_ALL_CANDIDATES_AND_OLD_WINNERS','candidates':1856,
                                          'old_winners':232,'max_scale_gap':max(x['scale_abs_gap'] for x in replay),
                                          'max_score_gap':max(x['score_abs_gap'] for x in replay),'new_outer_fits_required':required})
        if args.score_only:
            print(json.dumps({'status':'SCORING_COMPLETE','new_inner_fits':0,'new_outer_fits':0,'new_outer_fits_required':required}))
            return
        output=[]
        selected={(s['slot_id'],s['objective']):s for s in selections}
        fits=0
        for branch,slot in slots:
            ids,groups,y,oldraw=validate_population(test_data[slot['slot_id']],slot['slot_id'])
            for objective in OBJECTIVES:
                chosen=selected[(slot['slot_id'],objective)]
                raw=oldraw
                if chosen['requires_new_fit']:
                    result=branch.fit_predict(slot,chosen['candidate_id'],out/'models'/slot['slot_id']/objective)
                    raw=np.asarray(result['raw'],np.float64)
                    require(raw.shape==y.shape and np.isfinite(raw).all(),'Invalid new raw prediction')
                    fits+=1
                frame=pd.DataFrame({'pair_id':ids,'component_id':groups,'true_delta':y,'raw_prediction':raw,
                                    'scale_c':chosen['scale_c'],'calibrated_prediction':chosen['scale_c']*raw})
                for key in ('domain','slot_id','representation','pipeline','repeat','outer_fold'):
                    frame[key]=slot[key]
                frame['objective']=objective
                frame['candidate_id']=chosen['candidate_id']
                frame['absolute_error']=np.abs(frame.true_delta-frame.calibrated_prediction)
                output.append(frame)
        require(fits==required,'Fit accounting differs')
        predictions=pd.concat(output,ignore_index=True)
        predictions.to_csv(results/'OUTER_PREDICTIONS.csv',index=False,lineterminator='\n')
        summary=summarize(predictions,common,results)
        save_json(out/'RUN_SUMMARY.json',{'new_inner_fits':0,'new_outer_fits':fits,'summary':summary})
        print(json.dumps({'status':'COMPLETE','new_inner_fits':0,'new_outer_fits':fits,'prediction_rows':len(predictions)}))

if __name__=='__main__':
    main()
