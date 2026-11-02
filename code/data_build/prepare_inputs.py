"""Prepare model inputs from lawfully obtained final curated tables and fixed maps.

This command does not reconstruct the author's curation from the raw snapshot.
It checks input identities, computes Morgan counts and training-only vocabularies,
and writes a fresh local input bundle. It never reassigns identities to folds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['records', 'pairs', 'fingerprints', 'outer-folds', 'inner-folds']:
        parser.add_argument('--' + key, required=True, type=Path)
    for key in ['record-sources', 'selected-sources', 'source-inner-folds']:
        parser.add_argument('--' + key, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output must not already exist')
    prepare(args)


def prepare(args, *, verify_original_tables=True):
    if args.output.exists():
        raise RuntimeError('output must not already exist')
    expected = json.loads((HERE / 'expected_inputs.json').read_text('utf-8'))['fixed_input_identities']
    identities = {'records': 'input_build/data/records_revised.csv',
        'pairs': 'input_build/data/pairs_revised.csv', 'fingerprints': 'input_build/data/fingerprints.csv',
        'outer_folds': 'input_build/data/outer_identity_folds.csv',
        'inner_folds': 'input_build/config/inner_identity_folds.csv',
        'record_sources': 'input_build/data/record_sources.csv',
        'selected_sources': 'input_build/config/selected_sources.json',
        'source_inner_folds': 'input_build/config/source_inner_identity_folds.csv'}
    supplied = {key: getattr(args, key) for key in identities if getattr(args, key) is not None}
    for key, path in supplied.items():
        if verify_original_tables and digest(path) != expected[identities[key]]['sha256']:
            raise RuntimeError('input differs from the final frozen table: ' + key)
    from input_schema import inspect_inputs
    report = inspect_inputs(args.records, args.pairs, args.fingerprints, args.outer_folds, args.inner_folds)
    if (report['eligible_records'], report['pairs'], report['identities']) != (699, 5804, 511):
        raise RuntimeError('curated population differs from the manuscript')
    sys.path.insert(0, str(HERE.parent / 'core'))
    import numpy as np
    import pandas as pd
    from count_features import compute_counts, layer_name
    records = pd.read_csv(args.records).sort_values('record_id', kind='mergesort').reset_index(drop=True)
    fps = pd.read_csv(args.fingerprints)
    counts, _ = compute_counts(records, fps)
    outer, inner = pd.read_csv(args.outer_folds), pd.read_csv(args.inner_folds)
    output = args.output.resolve()
    output.mkdir(parents=True)
    files = {}
    names = {'records':'records_revised.csv', 'pairs':'pairs_revised.csv',
             'fingerprints':'fingerprints.csv', 'outer_folds':'outer_identity_folds.csv',
             'inner_folds':'inner_identity_folds.csv', 'source_inner_folds':'source_inner_identity_folds.csv',
             'record_sources':'record_sources.csv', 'selected_sources':'selected_sources.json'}
    for key, path in supplied.items():
        destination = output / names[key]
        shutil.copyfile(path, destination)
        files[key] = {'path': destination.name, 'sha256': digest(destination)}
    cache = output / 'main_raw_sparse_counts.json'
    write_json(cache, {'records': [{'record_id': rid, 'counts': sorted(c.items())}
                                 for rid, c in zip(records.record_id, counts)]})
    files['counts'] = {'path': cache.name, 'sha256': digest(cache)}
    vocab_root = output / 'vocab'
    vocab_root.mkdir()
    vocabularies = {}
    for repeat in range(3):
        om = outer.loc[outer['repeat'].eq(repeat)].set_index('identity_hash').outer_fold
        for fold in range(5):
            im = inner.loc[inner['repeat'].eq(repeat) & inner.outer_fold.eq(fold)].set_index('identity_hash').inner_fold
            for inner_fold in [-1, 0, 1, 2]:
                mapped = records.identity_hash.map(om if inner_fold == -1 else im)
                train = mapped.notna().to_numpy() & mapped.ne(fold if inner_fold == -1 else inner_fold).to_numpy()
                vocab = np.asarray(sorted({k for i in np.flatnonzero(train) for k in counts[i]}), dtype=np.uint64)
                name = layer_name(repeat, fold, inner_fold)
                path = vocab_root / (name + '.npz')
                np.savez_compressed(path, environment_ids=vocab,
                    training_record_ids=records.loc[train, 'record_id'].to_numpy(str),
                    training_identity_hashes=np.asarray(sorted(set(records.loc[train, 'identity_hash'])), dtype=str))
                vocabularies[name] = {'path': 'vocab/' + path.name, 'sha256': digest(path),
                                     'vocabulary_size': len(vocab)}
    index = output / 'vocab_index.json'
    write_json(index, vocabularies)
    files['vocab_index'] = {'path': index.name, 'sha256': digest(index)}
    # Every vocabulary is independently bound in addition to the index.
    for key, value in vocabularies.items():
        files['vocab_' + key] = {'path': value['path'], 'sha256': value['sha256']}
    write_json(output / 'inputs.json', {'schema': 'PREPARED_CURATED_INPUTS_V1', 'files': files,
        'scope': report, 'fold_assignment': 'supplied frozen maps; no new split',
        'raw_snapshot_reconstruction': 'not performed', 'fits': 0})
    print(json.dumps({'status': 'LOCAL_INPUTS_PREPARED', 'vocabulary_layers': len(vocabularies),
                      'fits': 0, 'manifest': str(output / 'inputs.json')}))


if __name__ == '__main__':
    main()
