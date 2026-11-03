"""Join fixed external cases, infer from local models, or summarize private labels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
CORE = HERE.parent / 'core'


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text('utf-8'))


def table(path):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest().upper()


def content_digest(values):
    return hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest().upper()


def truth(value):
    token = str(value).strip().lower()
    require(token in ('true', 'false', '1', '0'), 'invalid boolean')
    return token in ('true', '1')


def mappings(case):
    records = table(HERE / (case.lower() + '_compounds.csv'))
    pairs = table(HERE / (case.lower() + '_pairs.csv'))
    expected = (7, 21) if case == 'BRD4' else (10, 45)
    require((len(records), len(pairs)) == expected, 'case mapping census differs')
    require(len({r['record_id'] for r in records}) == len(records), 'duplicate mapped record')
    by_id = {r['record_id']: r for r in records}
    require(len({p['pair_id'] for p in pairs}) == len(pairs), 'duplicate mapped pair')
    for pair in pairs:
        for suffix in ('i', 'j'):
            record = by_id[pair['record_id_' + suffix]]
            require(record['identity_hash'] == pair['identity_hash_' + suffix], 'endpoint identity differs')
            require(record['compound_id'] == pair['compound_' + suffix], 'compound correspondence differs')
        require(pair['prediction_direction'] == 'pDC50_j_minus_pDC50_i', 'direction differs')
    require(sum(truth(p['primary_ge05']) for p in pairs) == (21 if case == 'BRD4' else 42), 'primary census differs')
    return records, pairs


def member_manifest():
    result = read_json(HERE / 'member_manifest.json')
    require(result['schema'] == 'FIXED_EXTERNAL_MEMBERS_V1', 'unknown member schema')
    members = result['members']
    require(len(members) == 420 and len({m['member_id'] for m in members}) == 420, 'member census differs')
    return result


def scientific_imports():
    # Importing these retained helpers does not fit or load any estimator.
    sys.path.insert(0, str(CORE))
    import portable as core_portable
    cfg = read_json(CORE / 'scientific_config.json')
    core_portable.check_versions(cfg)
    import numpy as np
    import pandas as pd
    import run_binary
    import run_count
    import count_features
    return np, pd, run_binary, run_count, count_features


def case_features(case, path, np, pd, binary, count_features):
    from rdkit import Chem
    mapped_records, mapped_pairs = mappings(case)
    supplied = table(path)
    by_compound = {r['compound_id']: r for r in supplied}
    require(len(supplied) == len(by_compound) == len(mapped_records), 'private compound count differs')
    require(set(by_compound) == {r['compound_id'] for r in mapped_records}, 'private compound numbers differ')
    gen = count_features.generator()
    records, fingerprints = [], []
    for record in sorted(mapped_records, key=lambda r: r['record_id']):
        smiles = by_compound[record['compound_id']]['canonical_isomeric_smiles_full']
        mol = Chem.MolFromSmiles(smiles)
        require(mol is not None, 'cannot parse private structure: ' + record['compound_id'])
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        require(hashlib.sha256(canonical.encode()).hexdigest().upper() == record['identity_hash'],
                'private full graph differs from retained identity: ' + record['compound_id'])
        records.append(record | {'canonical_isomeric_smiles_full': canonical})
        bits = gen.GetFingerprint(mol).GetOnBits()
        fingerprints.append({'record_id': record['record_id'], 'fp_hex': format(sum(1 << b for b in bits), '0512X')})
    records = pd.DataFrame(records)
    fps = pd.DataFrame(fingerprints)
    counts, _ = count_features.compute_counts(records, fps)
    X = np.vstack([binary.decode_fp(row['fp_hex']) for row in fingerprints])
    pairs = pd.DataFrame(mapped_pairs).sort_values('pair_id', kind='mergesort').reset_index(drop=True)
    index = {rid: i for i, rid in enumerate(records.record_id)}
    pi = np.array([index[rid] for rid in pairs.record_id_i], dtype=np.int64)
    pj = np.array([index[rid] for rid in pairs.record_id_j], dtype=np.int64)
    if case == 'RIPK1':
        ints = [int(row['fp_hex'], 16) for row in fingerprints]
        observed = [((ints[i] & ints[j]).bit_count() / (ints[i] | ints[j]).bit_count()) >= .5 - 1e-12
                    for i, j in zip(pi, pj)]
        require(observed == [truth(p) for p in pairs.primary_ge05], 'private graph primary membership differs')
    return SimpleNamespace(records=records, pairs=pairs, X_record=X, raw_counts=counts, pair_i=pi, pair_j=pj,
        X_diff=count_features.PairFeatures(X, pi, pj, 'difference'), X_sum=count_features.PairFeatures(X, pi, pj, 'sum'))


def member_model_path(args, member):
    cache = args.cache.resolve()
    if args.mode == 'original':
        path = cache / member['cache_model_path']
        require(digest(path) == member['model_sha256'], 'original cached model byte identity differs')
        return path
    run = cache / member['portable_run_directory']
    binding = read_json(run / 'INPUT_BINDING.json')
    require(binding['manifest_sha256'] == digest(args.inputs), 'rebuilt unit input binding differs')
    require(binding['config_sha256'] == digest(CORE / 'scientific_config.json'), 'rebuilt unit scientific configuration differs')
    expected = {'command': 'run-fold', 'representation': member['representation'], 'learner': member['learner'],
                'repeat': member['repeat'], 'fold': member['outer_fold'], 'pipeline': member['strategy']}
    require(all(binding['arguments'].get(k) == v for k, v in expected.items()), 'rebuilt unit member identity differs')
    cell = run / 'results' / f"{member['learner']}_R{member['repeat']}" / f"outer_{member['outer_fold']}" / member['strategy']
    selected = read_json(cell / 'selection.json')['selected']
    require(selected['candidate_id'] == member['selected_candidate_id'], 'rebuilt candidate differs from retained member')
    require(selected['params'] == member['selected_params'], 'rebuilt selected parameters differ')
    require(abs(selected['scale_c'] - member['scale_c']) <= 1e-12, 'rebuilt inner-OOF calibration differs')
    path = Path(read_json(cell / 'final/unit_complete.json')['model_path']).resolve(strict=True)
    require(path.is_relative_to(run.resolve()), 'rebuilt final model is outside its bound unit')
    return path


def vocabulary(args, member, manifest, prepared, np):
    item = manifest['outer_vocabularies'][member['vocabulary_layer']]
    path = args.cache / item['cache_path'] if args.mode == 'original' else (
        Path(prepared['files']['vocab_index']['absolute_path']).parent / 'vocab' / (item['layer'] + '.npz'))
    if args.mode == 'original':
        require(digest(path) == item['sha256'], 'original vocabulary byte identity differs')
    with np.load(path, allow_pickle=False) as values:
        ids, training = values['environment_ids'].tolist(), values['training_record_ids'].tolist()
    require(len(ids) == item['size'] and len(training) == item['training_records'], 'vocabulary support differs')
    require(content_digest(ids) == item['environment_ids_sha256'], 'vocabulary column IDs/order differ')
    require(content_digest(training) == item['training_record_ids_sha256'], 'vocabulary training member/order differs')
    return np.asarray(ids)


def infer(args):
    np, pd, binary, count, features = scientific_imports()
    import joblib
    import portable as core_portable
    import model_adapters
    import ripk1_statistics
    manifest = member_manifest()
    prepared = None
    if args.mode == 'rebuilt':
        require(args.inputs is not None, 'rebuilt mode requires --inputs')
        prepared = core_portable.load_manifest(args.inputs)
        core_portable.inspect(prepared)
    base = case_features(args.case, args.compounds, np, pd, binary, features)
    rows = []
    for member in manifest['members']:
        current = base
        if member['representation'] == 'count':
            vocab = vocabulary(args, member, manifest, prepared, np)
            X, _ = features.map_counts(base.raw_counts, vocab)
            current = SimpleNamespace(**vars(base))
            current.X_record = X
            current.X_diff = features.PairFeatures(X, base.pair_i, base.pair_j, 'difference')
            current.X_sum = features.PairFeatures(X, base.pair_i, base.pair_j, 'sum')
        path = member_model_path(args, member)
        model_sha = digest(path)
        model = joblib.load(path)
        ripk1_statistics.validate_model(model, member)
        if member['learner'] == 'xgboost':
            device, _ = model_adapters._xgboost_config_device(model)
            require(device in ('cuda', 'cuda:0'), 'XGBoost requires retained CUDA ordinal 0; no CPU fallback or other ordinal')
        runner = binary if member['representation'] == 'binary' else count
        raw = runner.predict_pipeline(model, member['strategy'], current, np.arange(len(base.pairs)))
        if member['learner'] == 'xgboost':
            device, _ = model_adapters._xgboost_config_device(model)
            require(device in ('cuda', 'cuda:0'), 'XGBoost fell back from CUDA ordinal 0')
        calibrated = member['scale_c'] * raw
        require(np.isfinite(calibrated).all(), 'nonfinite calibrated prediction')
        for i, pair in base.pairs.iterrows():
            rows.append({**{key: member[key] for key in ['representation', 'learner', 'strategy', 'pipeline', 'combo_id',
                'member_id', 'repeat', 'outer_fold', 'selected_candidate_id', 'scale_c', 'ensemble_weight']},
                'model_sha256': model_sha, 'pair_id': pair.pair_id, 'record_id_i': pair.record_id_i,
                'record_id_j': pair.record_id_j, 'primary_ge05': truth(pair.primary_ge05),
                'raw_prediction': float(raw[i]), 'calibrated_prediction': float(calibrated[i])})
        del model
    args.output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(args.output / 'member_predictions.csv', index=False, lineterminator='\n')
    print(json.dumps({'case': args.case, 'members': 420, 'rows': len(rows), 'fits': 0, 'cache_mode': args.mode}))


def summarize(args):
    np, pd, _, _, _ = scientific_imports()
    mapped_records, mapped_pairs = mappings(args.case)
    measured = table(args.measurements)
    concentrations = {r['compound_id']: float(r['DC50_nM']) for r in measured}
    require(len(measured) == len(concentrations) == len(mapped_records), 'private measurement count differs')
    require(set(concentrations) == {r['compound_id'] for r in mapped_records}, 'private measurement compound numbers differ')
    require(all(np.isfinite(v) and v > 0 for v in concentrations.values()), 'invalid private DC50 concentration')
    z = {r['record_id']: 9 - np.log10(concentrations[r['compound_id']]) for r in mapped_records}
    pairs = pd.DataFrame(mapped_pairs).sort_values('pair_id', kind='mergesort').reset_index(drop=True)
    pairs['primary_ge05'] = pairs.primary_ge05.map(truth)
    pairs['true_delta'] = [z[r.record_id_j] - z[r.record_id_i] for r in pairs.itertuples()]
    members = pd.read_csv(args.members)
    require('true_delta' not in members, 'use label-free member predictions; labels enter only through --measurements')
    # Enforce the fixed ID joins before any metric calculation.
    endpoint_map = pairs.set_index('pair_id')
    require(set(members.pair_id) == set(endpoint_map.index), 'prediction pair set differs')
    for suffix in ('i', 'j'):
        column = 'record_id_' + suffix
        require(np.array_equal(members[column], members.pair_id.map(endpoint_map[column])), 'prediction endpoint/direction differs')
    require(np.allclose(members.raw_prediction * members.scale_c, members.calibrated_prediction,
                        rtol=1e-12, atol=1e-12), 'member calibration differs')
    if 'strategy' not in members:
        members['strategy'] = members.pipeline
    if 'pipeline' not in members:
        members['pipeline'] = members.strategy
    if 'member_id' not in members:
        members['member_id'] = [f'{r.representation}|{r.learner}|R{r.repeat}|O{r.outer_fold}|{r.strategy}' for r in members.itertuples()]
    frozen = {m['member_id']: m for m in member_manifest()['members']}
    require(set(members.member_id) == set(frozen), 'fixed member set differs')
    require(not members.duplicated(['member_id', 'pair_id']).any() and len(members) == 420 * len(pairs), 'fixed member-pair population differs')
    for r in members.itertuples():
        m = frozen[r.member_id]
        require(all(getattr(r, k) == m[k] for k in ['representation', 'learner', 'strategy', 'repeat', 'outer_fold', 'selected_candidate_id']), 'member metadata differs')
        require(abs(r.scale_c - m['scale_c']) <= 1e-12, 'fixed member scale differs')
    members['true_delta'] = members.pair_id.map(endpoint_map.true_delta)
    if args.case == 'BRD4':
        import brd4_statistics
        ensemble, metric = brd4_statistics.summarize_external(members)
        outputs = {'ensemble_predictions.csv': ensemble, 'ensemble_metrics.csv': metric}
    else:
        import ripk1_statistics
        members['primary_ge05'] = members.pair_id.map(endpoint_map.primary_ge05)
        ensemble, metric, contrast = ripk1_statistics.summarize_members(members, pairs, pairs.primary_ge05.to_numpy(),
            {'binary': SimpleNamespace(y_pair=pairs.true_delta.to_numpy(float))})
        outputs = {'ensemble_predictions.csv': ensemble, 'ensemble_metrics.csv': metric, 'contrasts.csv': contrast}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, frame in outputs.items():
        frame.to_csv(args.output / name, index=False, lineterminator='\n')
    print(json.dumps({'case': args.case, 'point_estimates_only': True, 'fits': 0, 'resampling': 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    inspection = sub.add_parser('inspect', help='inspect public case IDs and model metadata; no scientific imports')
    inference = sub.add_parser('infer', help='predict from local retained or publicly rebuilt identity-CV model cache')
    summary = sub.add_parser('summarize', help='compute retained case point estimates from private measured labels')
    for command in (inspection, inference, summary):
        command.add_argument('--case', required=True, choices=['BRD4', 'RIPK1'])
    inference.add_argument('--compounds', required=True, type=Path)
    inference.add_argument('--cache', required=True, type=Path)
    inference.add_argument('--mode', required=True, choices=['original', 'rebuilt'])
    inference.add_argument('--inputs', type=Path)
    summary.add_argument('--members', required=True, type=Path)
    summary.add_argument('--measurements', required=True, type=Path)
    for command in (inference, summary):
        command.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'inspect':
        records, pairs = mappings(args.case)
        m = member_manifest()
        print(json.dumps({'case': args.case, 'records': len(records), 'pairs': len(pairs),
            'primary_pairs': sum(truth(p['primary_ge05']) for p in pairs), 'shared_members': len(m['members']),
            'combinations': m['model_combinations'], 'fits': 0, 'model_imports': 0}))
    elif args.command == 'infer':
        infer(args)
    else:
        summarize(args)


if __name__ == '__main__':
    main()
