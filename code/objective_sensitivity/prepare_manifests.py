"""Index locally generated public CLI caches without training or prediction."""
from pathlib import Path
import argparse
import json

def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',required=True,type=Path)
    parser.add_argument('--internal-runs',required=True,type=Path,help='Parent containing 60 core portable.py run-fold output directories')
    parser.add_argument('--loco-runs',required=True,type=Path,help='Parent containing 172 Ridge E1/matched-E0 run_loco.py output directories')
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    out=args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory must be empty')
    out.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for path in sorted(args.internal_runs.rglob('INPUT_BINDING.json')):
        a=json.loads(path.read_text(encoding='utf-8'))['arguments']
        if a.get('command')!='run-fold' or a.get('learner')!='ridge' or a.get('pipeline') not in ('endpoint_E1','endpoint_E0_matched'):
            continue
        cell=path.parent/'results'/f"ridge_R{a['repeat']}"/f"outer_{a['fold']}"/a['pipeline']
        if not (cell/'selection.json').is_file():
            raise ValueError('Incomplete identity cache cell')
        jobs.append({'learner':'ridge','representation':a['representation'],'pipeline':a['pipeline'],
                     'repeat':int(a['repeat']),'outer_fold':int(a['fold']),'cell_path':str(cell.resolve())})
    keys={(j['representation'],j['pipeline'],j['repeat'],j['outer_fold']) for j in jobs}
    expected={(rep,p,repeat,fold) for rep in ('binary','count') for p in ('endpoint_E1','endpoint_E0_matched')
              for repeat in range(3) for fold in range(5)}
    if len(jobs)!=60 or keys!=expected:
        raise ValueError('Expected exactly 60 complete, unique identity cache cells')
    jobs.sort(key=lambda j:(j['representation'],j['repeat'],j['outer_fold'],j['pipeline']))
    slots=[]
    for path in sorted(args.loco_runs.rglob('RUN_SUMMARY.json')):
        a=json.loads(path.read_text(encoding='utf-8'))
        if a.get('learner')!='ridge' or a.get('pipeline') not in ('endpoint_E1','endpoint_E0_matched'):
            continue
        selection=json.loads((path.parent/'selection'/a['representation']/(a['pipeline']+'.json')).read_text(encoding='utf-8'))
        selected=selection['selected']
        slots.append({'outer_fold':a['component'],'representation':a['representation'],'pipeline':a['pipeline'],
                      'cell':str(path.parent.resolve()),'test_predictions':str((path.parent/'OUTER_PREDICTIONS.csv').resolve()),
                      'old_candidate_id':selected['candidate_id'],'old_scale':selected['scale_c'],
                      'old_scores':{r['candidate_id']:{'scale_c':r['scale_c'],'selection_score':r['selection_score']}
                                    for r in selection['candidates']},'oof_format':'csv'})
    keys={(j['outer_fold'],j['representation'],j['pipeline']) for j in slots}
    groups={j['outer_fold'] for j in slots}
    if len(slots)!=172 or len(groups)!=43 or keys!={(g,r,p) for g in groups for r in ('binary','count') for p in ('endpoint_E1','endpoint_E0_matched')}:
        raise ValueError('Expected exactly 172 complete, unique component cache cells')
    write(out/'internal.json',{'data_dir':str(args.data_dir.resolve()),'jobs':jobs})
    write(out/'components.json',{'data_dir':str(args.data_dir.resolve()),'slots':slots})
    write(out/'sensitivity.json',{'internal_manifest':'internal.json','loco_manifest':'components.json'})
    print(json.dumps({'status':'CACHE_MANIFESTS_PREPARED','internal_slots':60,'component_slots':172,'fits':0,'predictions':0}))

if __name__=='__main__':
    main()
