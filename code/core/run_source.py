"""Frozen S0/S1/S3 source evaluation. Selection, seeds and model math retained."""
from pathlib import Path
import json,os,time
import numpy as np
import pandas as pd
from run_binary import (candidate_grid,fit_unit,FitLedger,nonnegative_l1_scale,atomic_json,atomic_npz,atomic_csv,sha256,now_utc,count_ledger_events)
ROOT=None
def masks(bundle,source,s):
 rtrain=source!=s; rtest=~rtrain
 ptrain=rtrain[bundle.pair_i]&rtrain[bundle.pair_j]
 ptest=rtest[bundle.pair_i]&rtest[bundle.pair_j]
 trainids=set(bundle.records.loc[rtrain,'identity_hash'])
 unseen_i=~bundle.pairs.identity_hash_i.isin(trainids).to_numpy()
 unseen_j=~bundle.pairs.identity_hash_j.isin(trainids).to_numpy()
 return rtrain,ptrain,ptest,unseen_i,unseen_j

def metrics(y,p):
 nonzero=np.abs(y)>1e-12
 predicted_zero=np.abs(p)<=1e-12
 sign_right=(np.sign(p)==np.sign(y))&~predicted_zero
 return dict(n=len(y),mae=float(np.mean(np.abs(y-p))),rmse=float(np.sqrt(np.mean((y-p)**2))),direction_nonzero_truth_n=int(nonzero.sum()),direction_accuracy_strict=float(np.mean(sign_right[nonzero])) if nonzero.any() else None,direction_accuracy_zero_half=float(np.mean(np.where(predicted_zero,.5,sign_right)[nonzero])) if nonzero.any() else None,predicted_zero_n=int(predicted_zero.sum()),true_zero_n=int((~nonzero).sum()))

def run(learner,cfg,b,spec,source,root,inner_path):
 global ROOT
 ROOT=Path(root)
 batch=f'C2_{learner}';outroot=ROOT/'results'/learner
 ledgerpath=ROOT/'logs'/learner/'fit_ledger.jsonl'
 ledger=FitLedger(ledgerpath,dict(learner=learner,model_seed=spec['model_seed'],analysis='C2_prespecified_source_holdout'))
 statuspath=ROOT/'runtime'/f'{learner}_status.json'
 started=time.time();context={}
 def status(extra):
  atomic_json(statuspath,dict(status='RUNNING',learner=learner,pid=os.getpid(),cwd=Path.cwd().as_posix(),started_at_utc=now_utc(),elapsed_seconds=time.time()-started,ledger_events=count_ledger_events(ledgerpath),**extra))
 pipelines=['direct_pair','endpoint_E0','endpoint_E1']+([] if learner=='ridge' else ['context_pair_C1'])
 innerall=pd.read_csv(inner_path)
 try:
  for x in spec['source_holdouts']:
   k=int(x['source_index'])
   rt,pt,pe,ui,uj=masks(b,source,x['source_id']);testidx=np.flatnonzero(pe)
   inner=innerall[innerall.source_index.eq(k)];fmap=dict(zip(inner.identity_hash,inner.inner_fold))
   ri=b.records.identity_hash.map(fmap).to_numpy();fi=b.pairs.identity_hash_i.map(fmap).to_numpy();fj=b.pairs.identity_hash_j.map(fmap).to_numpy()
   for pipeline in pipelines:
    out=outroot/x['holdout_id']/pipeline
    if (out/'summary.json').exists():continue
    candidates=[];common=None
    for candidate in candidate_grid(cfg,learner):
     cid=candidate['candidate_id'];cr=out/'candidates'/cid
     complete=cr/'candidate_complete.json';saved=cr/'oof_predictions.npz'
     if complete.exists():
      summary=json.loads(complete.read_text(encoding='utf-8'))
      with np.load(saved,allow_pickle=False) as z:idx,raw=z['pair_indices'],z['raw_prediction']
     else:
      idxs=[];preds=[];fitinfos=[]
      for f in range(3):
       rtrain=rt&(ri!=f);ptrain=pt&(fi!=f)&(fj!=f);idx=np.flatnonzero(pt&(fi==f)&(fj==f))
       if not len(idx) or not ptrain.any():raise RuntimeError('empty inner split')
       context=dict(batch_id=batch,outer_fold=k,pipeline=pipeline,phase='inner',candidate_id=cid,inner_fold=f,heldout_source=x['source_id'])
       status(dict(stage='fit_unit',**context))
       raw,info,_=fit_unit(learner=learner,pipeline=pipeline,params=candidate['params'],seed=spec['model_seed'],bundle=b,record_train=rtrain,pair_train=ptrain,predict_indices=idx,unit=cr/f'inner_{f}',ledger=ledger,context=context)
       idxs.append(idx);preds.append(raw);fitinfos.append(dict(inner_fold=f,validation_pairs=len(idx),fit_count=info['fit_count']))
      idx=np.concatenate(idxs);raw=np.concatenate(preds);order=np.argsort(idx,kind='mergesort');idx,raw=idx[order],raw[order]
      if len(np.unique(idx))!=len(idx):raise RuntimeError('duplicate OOF pair')
      c=nonnegative_l1_scale(b.y_pair[idx],raw)
      summary=dict(candidate_id=cid,params=candidate['params'],inner_oof_pairs=len(idx),scale_c=c,inner_oof_raw_mae=float(np.mean(np.abs(b.y_pair[idx]-raw))),inner_oof_calibrated_mae=float(np.mean(np.abs(b.y_pair[idx]-c*raw))),fold_fit_info=fitinfos)
      atomic_npz(saved,pair_indices=idx,raw_prediction=raw);atomic_json(complete,summary)
     if common is None:common=idx
     elif not np.array_equal(common,idx):raise RuntimeError('OOF population mismatch')
     candidates.append(summary)
    best=min(x['inner_oof_calibrated_mae'] for x in candidates)
    chosen=min((x for x in candidates if x['inner_oof_calibrated_mae']<=best+1e-12),key=lambda x:x['candidate_id'])
    atomic_json(out/'selection.json',dict(selected=chosen,candidates=candidates,heldout_source=x['source_id'],heldout_source_used_for_selection=False))
    context=dict(batch_id=batch,outer_fold=k,pipeline=pipeline,phase='source_final',candidate_id=chosen['candidate_id'],inner_fold=-1,heldout_source=x['source_id'],model_output_root=(ROOT/'models').as_posix())
    status(dict(stage='final_fit',**context))
    raw,info,_=fit_unit(learner=learner,pipeline=pipeline,params=chosen['params'],seed=spec['model_seed'],bundle=b,record_train=rt,pair_train=pt,predict_indices=testidx,unit=out/'final',ledger=ledger,context=context)
    c=float(chosen['scale_c']);test=b.pairs.iloc[testidx]
    train_targets=set(b.records.loc[rt,'normalized_poi_id'])
    frame=test[['pair_id','record_id_i','record_id_j','identity_hash_i','identity_hash_j','normalized_poi_id','pair_context_hash']].copy()
    frame['holdout_id']=x['holdout_id'];frame['heldout_source']=x['source_id'];frame['learner']=learner;frame['pipeline']=pipeline
    frame['true_delta']=b.y_pair[testidx];frame['raw_prediction']=raw;frame['calibrated_prediction']=c*raw;frame['scale_c']=c
    frame['identity_unseen_i']=ui[testidx];frame['identity_unseen_j']=uj[testidx];frame['both_identity_unseen']=ui[testidx]&uj[testidx]
    frame['identity_scenario']=np.where(ui[testidx]&uj[testidx],'both_unseen',np.where(ui[testidx]^uj[testidx],'one_seen','both_seen'))
    frame['target_seen_in_train']=test.normalized_poi_id.isin(train_targets).to_numpy();frame['selected_candidate_id']=chosen['candidate_id']
    atomic_csv(out/'source_predictions.csv',frame)
    summary=dict(holdout_id=x['holdout_id'],heldout_source=x['source_id'],learner=learner,pipeline=pipeline,scale_c=c,selected_candidate_id=chosen['candidate_id'],source_unseen_test_edges=len(frame),both_identity_unseen_edges=int(frame.both_identity_unseen.sum()),train_records=int(rt.sum()),train_pairs=int(pt.sum()),final_fit=info,metrics=[])
    for scene,mask in [('source_unseen',np.ones(len(frame),bool)),('source_identity_unseen',frame.both_identity_unseen.to_numpy()),('one_identity_seen',frame.identity_scenario.eq('one_seen').to_numpy()),('both_identity_seen',frame.identity_scenario.eq('both_seen').to_numpy())]:
     if not mask.any():continue
     for variant,pred in [('raw',raw),('calibrated',c*raw),('zero',np.zeros(len(raw)))]:summary['metrics'].append(dict(scenario=scene,variant=variant,**metrics(b.y_pair[testidx][mask],pred[mask])))
    atomic_json(out/'summary.json',summary)
    print(f'COMPLETE {learner} {x["holdout_id"]} {pipeline} n={len(frame)}',flush=True)
  expected=225 if learner=='ridge' else 600
  counts=count_ledger_events(ledgerpath)
  if counts.get('FIT_COMPLETE',0)!=expected:raise RuntimeError(f'fit count drift {counts} expected {expected}')
  atomic_json(statuspath,dict(status='COMPLETE',learner=learner,pid=os.getpid(),expected_fits=expected,ledger_events=counts,completed_at_utc=now_utc(),elapsed_seconds=time.time()-started))
  print(f'C2 {learner} COMPLETE fits={expected}',flush=True)
 except BaseException as exc:
  atomic_json(statuspath,dict(status='FAILED',learner=learner,pid=os.getpid(),error_type=type(exc).__name__,error=str(exc),context=context,ledger_events=count_ledger_events(ledgerpath),updated_at_utc=now_utc()))
  raise
