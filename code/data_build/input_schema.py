"""Original prepared-input schema checks; no model execution."""
from pathlib import Path
import csv
import re

class GateError(RuntimeError):
    pass

def require(condition, message):
    if not condition:
        raise GateError(message)

def read_table(path, required):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        require(set(required) <= set(reader.fieldnames or []), f'missing columns in {Path(path).name}: {required}')
        return list(reader)

def inspect_inputs(records_path, pairs_path, fingerprints_path, outer_path=None, inner_path=None):
    """Read identifiers/structure schema only; never recompute y, labels or metrics."""
    records = read_table(records_path, ['record_id','identity_hash','normalized_poi_id','pair_context_hash','pdc50_median'])
    if records and 'primary_pair_eligible' in records[0]:
        require(all(r['primary_pair_eligible'].lower() in ('true','false','1','0') for r in records), 'invalid eligibility flag')
        records = [r for r in records if r['primary_pair_eligible'].lower() in ('true','1')]
    by_id = {r['record_id']:r for r in records}
    require(len(by_id)==len(records) and bool(records), 'empty/duplicate eligible records')
    require(all(re.fullmatch('[0-9A-F]{64}',r['identity_hash']) for r in records), 'invalid identity key')
    pairs = read_table(pairs_path, ['pair_id','record_id_i','record_id_j','identity_hash_i','identity_hash_j','true_delta','is_cliff'])
    require(len({p['pair_id'] for p in pairs})==len(pairs) and bool(pairs), 'empty/duplicate pair set')
    for p in pairs:
        require(p['record_id_i'] in by_id and p['record_id_j'] in by_id, 'pair endpoint outside input')
        a,b=by_id[p['record_id_i']],by_id[p['record_id_j']]
        require(a['identity_hash']!=b['identity_hash'], 'same-identity pair')
        require(a['identity_hash']==p['identity_hash_i'] and b['identity_hash']==p['identity_hash_j'], 'pair identity mismatch')
        require(all(a[k] and a[k]==b[k] for k in ('normalized_poi_id','pair_context_hash')), 'pair condition mismatch')
    fps=read_table(fingerprints_path,['record_id','fp_hex'])
    fm={f['record_id']:f['fp_hex'] for f in fps}
    require(len(fm)==len(fps) and set(by_id)<=set(fm), 'fingerprint key coverage failure')
    require(all(re.fullmatch('[0-9A-F]{512}',fm[k]) for k in by_id), 'invalid binary fingerprint encoding')
    report={'eligible_records':len(records),'identities':len({r['identity_hash'] for r in records}),
            'pairs':len(pairs),'POI_ids':len({r['normalized_poi_id'] for r in records}),
            'label_values_transformed':False,'predictions_or_metrics_computed':False}
    require(bool(outer_path)==bool(inner_path), 'supply both split tables or neither')
    if outer_path:
        outer=read_table(outer_path,['repeat','identity_hash','outer_fold'])
        inner=read_table(inner_path,['repeat','outer_fold','identity_hash','inner_fold'])
        identities={r['identity_hash'] for r in records}
        require({int(r['repeat']) for r in outer}=={0,1,2}, 'outer repeats are not exactly 0,1,2')
        require({(int(r['repeat']),int(r['outer_fold'])) for r in inner}=={(r,o) for r in range(3) for o in range(5)}, 'inner cell coverage failure')
        for repeat in range(3):
            rows=[r for r in outer if int(r['repeat'])==repeat]
            om={r['identity_hash']:int(r['outer_fold']) for r in rows}
            require(len(rows)==len(om) and set(om)==identities and set(om.values())==set(range(5)), 'outer identity assignment failure')
            for fold in range(5):
                rows=[r for r in inner if int(r['repeat'])==repeat and int(r['outer_fold'])==fold]
                im={r['identity_hash']:int(r['inner_fold']) for r in rows}
                train={i for i,f in om.items() if f!=fold}
                require(len(rows)==len(im) and set(im)==train and set(im.values())=={0,1,2}, 'inner map is not exact outer-train scope')
        report['split_identity_scope']='3x5 outer and all inner maps checked; no new assignment'
    return report
