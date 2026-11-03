"""Inspect prepared inputs or explicitly run one fixed sensitivity-analysis unit."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def load_manifest(path):
    path = Path(path).resolve(strict=True)
    manifest = json.loads(path.read_text('utf-8'))
    if manifest.get('schema') != 'PREPARED_CURATED_INPUTS_V1':
        raise ValueError('unsupported prepared-input manifest')
    for key, item in manifest['files'].items():
        target = (path.parent / item['path']).resolve(strict=True)
        if not target.is_relative_to(path.parent) or digest(target) != item['sha256']:
            raise ValueError(f'prepared input identity mismatch: {key}')
        item['absolute_path'] = target.as_posix()
    return manifest


def bind_config(manifest):
    cfg = json.loads((HERE / 'scientific_config.json').read_text('utf-8'))
    cfg['inputs'] = {key: {'path': manifest['files'][key]['absolute_path'],
                          'sha256': manifest['files'][key]['sha256']}
                     for key in ('records', 'pairs', 'fingerprints')}
    cfg['count_inputs'] = {'main': {'path': manifest['files']['counts']['absolute_path'],
                                   'sha256': manifest['files']['counts']['sha256']}}
    return cfg


def check_versions(cfg):
    actual = {'python': '.'.join(map(str, sys.version_info[:3]))}
    for name in ('numpy', 'pandas', 'scikit_learn', 'xgboost', 'rdkit'):
        actual[name] = version(name.replace('_', '-'))
    expected = cfg['execution']['versions']
    comparable = dict(actual)
    # RDKit distribution metadata omits the leading zero in the release month.
    rdkit_actual = actual['rdkit'].split('.')
    rdkit_expected = expected['rdkit'].split('.')
    if (all(p.isdigit() for p in rdkit_actual + rdkit_expected)
            and tuple(map(int, rdkit_actual)) == tuple(map(int, rdkit_expected))):
        comparable['rdkit'] = expected['rdkit']
    if comparable != expected:
        raise RuntimeError('frozen scientific environment differs: ' + json.dumps(actual))


def inspect(manifest):
    sys.path.insert(0, str(HERE.parent / 'data_build'))
    from input_schema import inspect_inputs
    files = manifest['files']
    report = inspect_inputs(*(Path(files[k]['absolute_path']) for k in
                              ('records', 'pairs', 'fingerprints', 'outer_folds', 'inner_folds')))
    if (report['eligible_records'], report['pairs'], report['identities']) != (699, 5804, 511):
        raise ValueError('prepared inputs differ from the final manuscript population')
    return report


def run_fold(args, manifest, cfg):
    import pandas as pd
    import run_binary
    import run_count
    import count_features

    runner = run_binary if args.representation == 'binary' else run_count
    runner.ROOT = args.output.resolve()
    count_features.VOCAB_ROOT = Path(manifest['files']['vocab_index']['absolute_path']).parent / 'vocab'
    bundle = runner.load_bundle(cfg)
    outer = pd.read_csv(manifest['files']['outer_folds']['absolute_path'])
    inner = pd.read_csv(manifest['files']['inner_folds']['absolute_path'])
    outer = outer.loc[outer['repeat'].eq(args.repeat)]
    inner = inner.loc[inner['repeat'].eq(args.repeat) & inner.outer_fold.eq(args.fold)]
    outer_map = dict(zip(outer.identity_hash, outer.outer_fold))
    inner_map = dict(zip(inner.identity_hash, inner.inner_fold))
    identities = set(bundle.records.identity_hash)
    if set(outer_map) != identities or set(outer_map.values()) != set(range(5)):
        raise ValueError('outer identity map is not an exact five-fold cover of the eligible identities')
    outer_train = {identity for identity, assigned in outer_map.items() if assigned != args.fold}
    if set(inner_map) != outer_train or set(inner_map.values()) != {0, 1, 2}:
        raise ValueError('inner identity map is not the exact outer-train scope')
    cell = runner.ROOT / 'results' / f'{args.learner}_R{args.repeat}' / f'outer_{args.fold}'
    ledger_path = runner.ROOT / 'logs' / 'fits.jsonl'
    ledger = runner.FitLedger(ledger_path, {'representation': args.representation,
        'learner': args.learner, 'repeat': args.repeat, 'outer_fold': args.fold})
    summary = runner.run_pipeline(cfg=cfg, learner=args.learner, repeat=args.repeat,
        outer_fold=args.fold, bundle=bundle, external=None, outer_map=outer_map,
        inner_map=inner_map, pipeline=args.pipeline, cell_root=cell, ledger=ledger,
        seed=int(cfg['splitting']['model_seeds_paired_by_repeat'][args.repeat]),
        update_status=lambda value: runner.atomic_json(runner.ROOT / 'status.json', value))
    return {'status': 'UNIT_COMPLETE', 'summary': summary,
            'fit_events': runner.count_ledger_events(ledger_path)}


def run_sources(args, manifest, cfg):
    import numpy as np
    import pandas as pd
    import run_binary
    import run_source

    run_binary.ROOT = args.output.resolve()
    bundle = run_binary.load_bundle(cfg)
    sources = pd.read_csv(manifest['files']['record_sources']['absolute_path'],
                          keep_default_na=False).set_index('record_id')
    if set(sources.index) != set(bundle.records.record_id) or sources.index.duplicated().any():
        raise ValueError('source record scope mismatch')
    selection = json.loads(Path(manifest['files']['selected_sources']['absolute_path']).read_text('utf-8'))
    run_source.run(args.learner, cfg, bundle, selection,
                   bundle.records.record_id.map(sources.c2_source_id).to_numpy(),
                   args.output.resolve(), Path(manifest['files']['source_inner_folds']['absolute_path']))
    return {'status': 'SOURCE_UNITS_COMPLETE', 'representation': 'binary'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    inspection = sub.add_parser('inspect', help='check schemas and fixed member assignments; no model imports')
    inspection.add_argument('--manifest', required=True, type=Path)
    fold = sub.add_parser('run-fold', help='fit one fixed identity-CV sensitivity unit')
    source = sub.add_parser('run-source', help='fit the three retained source-holdout sensitivity analyses')
    for command in (fold, source):
        command.add_argument('--manifest', required=True, type=Path)
        command.add_argument('--learner', required=True, choices=['ridge', 'random_forest', 'xgboost'])
        command.add_argument('--output', required=True, type=Path)
    fold.add_argument('--representation', required=True, choices=['binary', 'count'])
    fold.add_argument('--repeat', required=True, type=int, choices=range(3))
    fold.add_argument('--fold', required=True, type=int, choices=range(5))
    fold.add_argument('--pipeline', required=True, choices=['direct_pair', 'endpoint_E0',
        'endpoint_E1', 'endpoint_E0_matched', 'context_pair_C1'])
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    report = inspect(manifest)
    if args.command == 'inspect':
        print(json.dumps({'status': 'INPUT_SCHEMA_ONLY_PASS', 'fits': 0, **report}))
        return
    cfg = bind_config(manifest)
    check_versions(cfg)
    if args.learner == 'ridge' and getattr(args, 'pipeline', None) == 'context_pair_C1':
        parser.error('context_pair_C1 is defined for tree learners only')
    if args.output.exists():
        parser.error('--output must be a new directory; the wrapper never resumes unbound checkpoints')
    if args.command == 'run-source' and not all(key in manifest['files'] for key in
                                               ['selected_sources', 'record_sources', 'source_inner_folds']):
        parser.error('source holdout requires all three optional source metadata inputs')
    args.output.mkdir(parents=True)
    (args.output / 'INPUT_BINDING.json').write_text(json.dumps({'manifest_sha256': digest(args.manifest),
        'config_sha256': digest(HERE / 'scientific_config.json'), 'arguments':
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}, indent=2) + '\n',
        encoding='utf-8', newline='\n')
    result = run_fold(args, manifest, cfg) if args.command == 'run-fold' else run_sources(args, manifest, cfg)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
