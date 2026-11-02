"""Install opaque source/fold assignments and build training-only vocabularies."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import numpy as np
import pandas as pd

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',required=True,type=Path)
    parser.add_argument('--component',help='Prepare a single pair-positive component; default prepares all 43')
    args=parser.parse_args()
    root=args.data_dir.resolve()
    maps=Path(__file__).with_name('mappings')
    records=pd.read_csv(root/'records_revised.csv').sort_values('record_id',kind='mergesort').reset_index(drop=True)
    pairs=pd.read_csv(root/'pairs_revised.csv').sort_values('pair_id',kind='mergesort').reset_index(drop=True)
    groups=pd.read_csv(maps/'record_source_group_map.csv')
    folds=pd.read_csv(maps/'inner_component_folds.csv')
    if len(records)!=699 or records.identity_hash.nunique()!=511 or len(pairs)!=5804:
        raise ValueError('Expected current 699-record/511-identity/5804-pair population')
    by_id=groups.set_index('record_id')
    if set(records.record_id)!=set(groups.record_id) or not np.array_equal(
            by_id.loc[records.record_id].identity_hash.to_numpy(),records.identity_hash.to_numpy()):
        raise ValueError('Opaque component assignments do not match reconstructed record identities')
    pair_groups=by_id.loc[pairs.record_id_i].component_id.to_numpy()
    if not np.array_equal(pair_groups,by_id.loc[pairs.record_id_j].component_id.to_numpy()):
        raise ValueError('Pair endpoints cross source components')
    heldouts=sorted(set(pair_groups))
    if len(heldouts)!=43:
        raise ValueError('Expected 43 pair-positive components')
    if args.component:
        if args.component not in heldouts:
            raise ValueError('Unknown pair-positive component')
        heldouts=[args.component]
    for name in ('record_source_group_map.csv','inner_component_folds.csv','pair_dependency_components.csv'):
        destination=root/name
        if destination.exists():
            if destination.read_bytes()!=(maps/name).read_bytes():
                raise ValueError('Refusing to replace existing different assignment file: '+name)
        else:
            shutil.copyfile(maps/name,destination)
    counts={row['record_id']:{int(k):int(v) for k,v in row['counts']}
            for row in json.loads((root/'main_raw_sparse_counts.json').read_text(encoding='utf-8'))['records']}
    rg=by_id.loc[records.record_id].component_id.to_numpy()
    vocab_dir=root/'vocab'
    vocab_dir.mkdir(exist_ok=True)
    manifest_path=root/'vocab_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    for heldout in heldouts:
        mapping=folds.loc[folds.outer_group_id.eq(heldout)].set_index('component_id').inner_fold.to_dict()
        rf=np.array([mapping.get(group,-1) for group in rg])
        if not np.all((rf<0)==(rg==heldout)) or set(rf)!={-1,0,1,2}:
            raise ValueError('Incomplete fixed component-fold assignment')
        for layer in (-1,0,1,2):
            key=f'R0_G{heldout}_I{layer}'
            ids=records.record_id.to_numpy(str)[(rf>=0) if layer==-1 else ((rf>=0)&(rf!=layer))]
            environments=np.array(sorted({environment for rid in ids for environment in counts[rid]}),dtype=np.uint64)
            destination=vocab_dir/(key+'.npz')
            if destination.exists():
                with np.load(destination,allow_pickle=False) as values:
                    if not np.array_equal(values['environment_ids'],environments) or not np.array_equal(values['training_record_ids'],ids):
                        raise ValueError('Existing vocabulary differs from fixed training-only union')
            else:
                np.savez_compressed(destination,environment_ids=environments,training_record_ids=ids)
            manifest[key]={'path':'vocab/'+destination.name,
                           'sha256':hashlib.sha256(destination.read_bytes()).hexdigest().upper(),
                           'training_scope':'all eligible layer training identities before singleton filter'}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'status':'COMPONENT_ASSIGNMENTS_AND_VOCABULARIES_PREPARED','components':len(heldouts),
                      'layers':4*len(heldouts),'fits':0,'predictions':0}))

if __name__=='__main__':
    main()
