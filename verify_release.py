"""Release integrity checks for this repository; standard library only, no model execution."""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json
import struct
import sys

LABEL_COLUMNS = {'true_delta', 'canonical_target_b_minus_a', 'absolute_error', 'tanimoto',
                 'pdc50', 'pdc50_median', 'dc50', 'dc50_nm', 'smiles',
                 'canonical_isomeric_smiles_full'}
UNDECLARED_ALLOWED = {'results/result_manifest.json', 'results/field_dictionary.csv'}

class GateError(RuntimeError):
    pass

def require(condition, message):
    if not condition:
        raise GateError(message)

def open_text(path):
    if path.name.endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8', newline='')
    return path.open(encoding='utf-8', newline='')

def read_header(path):
    with open_text(path) as handle:
        return next(csv.reader(handle))

def read_header_and_rows(path):
    with open_text(path) as handle:
        reader = csv.reader(handle)
        return next(reader), sum(1 for _ in reader)

def read_dicts(path):
    with open_text(path) as handle:
        return list(csv.DictReader(handle))

def iter_dicts(path):
    with open_text(path) as handle:
        for row in csv.DictReader(handle):
            yield row

def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha256_hex(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def payload_entries(repo):
    manifest = load_json(repo / 'results/result_manifest.json')
    require(manifest.get('schema_version') == 1, 'unexpected result manifest schema version')
    require(len(manifest['files']) == manifest['payload_files'], 'payload count differs from declared total')
    return manifest, manifest['files']

def check_payload(repo):
    manifest, entries = payload_entries(repo)
    for entry in entries:
        path = repo / entry['path']
        name = entry['path']
        require(path.is_file(), f'missing payload {name}')
        require(path.stat().st_size == entry['bytes'], f'byte size differs for {name}')
        if path.suffix == '.json':
            require(len(load_json(path)) == entry['rows'], f'element count differs for {name}')
            continue
        header, rows = read_header_and_rows(path)
        require(rows == entry['rows'], f'row count differs for {name}')
        require(header == entry['columns'], f'column list differs for {name}')
        require(not set(header) & set(entry.get('removed_columns') or []),
                f'declared-removed column still present in {name}')
        if entry.get('kind') == 'label_free_predictions':
            require(not {column.lower() for column in header} & LABEL_COLUMNS,
                    f'label column present in label-free payload {name}')
        if path.name.endswith('.gz'):
            require(struct.unpack('<I', path.read_bytes()[4:8])[0] == 0,
                    f'gzip header timestamp is not zero in {name}')
    on_disk = {p.relative_to(repo).as_posix() for p in (repo / 'results').rglob('*') if p.is_file()}
    undeclared = on_disk - {entry['path'] for entry in entries} - UNDECLARED_ALLOWED
    require(not undeclared, f'undeclared files under results/: {sorted(undeclared)}')
    return {'payload_files': len(entries), 'byte_row_column_exact': True,
            'gzip_timestamps_zero': True, 'label_columns_absent': True}

def check_curation(repo):
    base = repo / 'code/data_build/curation'
    index = load_json(base / 'index_manifest.json')
    for name, expected in index['files'].items():
        path = base / name
        require(path.is_file(), f'missing curation file {name}')
        require(sha256_hex(path).upper() == expected.upper(), f'checksum differs for {name}')
    records = read_dicts(base / 'record_index.csv')
    pairs = read_dicts(base / 'pair_index.csv')
    groups = read_dicts(repo / 'code/source_components/mappings/record_source_group_map.csv')
    pair_groups = read_dicts(repo / 'code/source_components/mappings/pair_dependency_components.csv')
    require(len({r['record_id'] for r in records}) == len(records), 'duplicate record_id in record index')
    require(len({p['pair_id'] for p in pairs}) == len(pairs), 'duplicate pair_id in pair index')
    require({p['pair_id'] for p in pairs} == {p['pair_id'] for p in pair_groups},
            'pair index and pair/component map cover different pairs')
    require({g['record_id'] for g in groups} == {r['record_id'] for r in records},
            'component map and record index cover different records')
    return {'checksums_verified': len(index['files']), 'records': len(records),
            'identities': len({r['identity_hash'] for r in records}), 'pairs': len(pairs),
            'components_indexed': len({g['component_id'] for g in groups}),
            'components_with_pairs': len({g['dependency_component'] for g in pair_groups}),
            'excluded_source_rows': len(load_json(base / 'excluded_source_rows.json')['raw_indices']),
            'graph_edit_rules': len(load_json(base / 'graph_edits.json')['rules']),
            'source_holdouts': len(load_json(base / 'selected_sources.json')['source_holdouts'])}

def check_declared_scope(repo, curation):
    population = load_json(repo / 'results/result_manifest.json')['input_population']
    scope = load_json(repo / 'data/source_manifest.json')['study_scope']
    expected = load_json(repo / 'code/data_build/expected_inputs.json')
    require(curation['records'] == population['records'] == scope['retained_records'] == expected['records'],
            'record count disagrees across manifests')
    require(curation['pairs'] == population['pairs'] == scope['retained_pairs'] == expected['compatible_pairs'],
            'pair count disagrees across manifests')
    require(curation['identities'] == population['identities'] == scope['complete_identity_count'] == expected['chemical_identities'],
            'identity count disagrees across manifests')
    require(curation['components_with_pairs'] == population['loco_components'] == scope['component_loco_groups_with_pairs'],
            'pair-positive component count disagrees across manifests')
    return {'sources_agreeing': 4, 'records': curation['records'], 'pairs': curation['pairs'],
            'identities': curation['identities'], 'components_with_pairs': curation['components_with_pairs']}

def pair_authority(repo):
    endpoints, identities = {}, {}
    for record in read_dicts(repo / 'code/data_build/curation/record_index.csv'):
        identities[record['record_id']] = record['identity_hash']
    for pair in read_dicts(repo / 'code/data_build/curation/pair_index.csv'):
        endpoints[pair['pair_id']] = (pair['record_id_i'], pair['record_id_j'])
    for name in ('brd4_pairs.csv', 'ripk1_pairs.csv'):
        for pair in read_dicts(repo / 'code/external_cases' / name):
            require(pair['pair_id'] not in endpoints, f"case pair_id collides with main index: {pair['pair_id']}")
            endpoints[pair['pair_id']] = (pair['record_id_i'], pair['record_id_j'])
            identities.setdefault(pair['record_id_i'], pair['identity_hash_i'])
            identities.setdefault(pair['record_id_j'], pair['identity_hash_j'])
    return endpoints, identities

def check_predictions(repo):
    endpoints, identities = pair_authority(repo)
    scope = load_json(repo / 'data/source_manifest.json')['study_scope']
    checked = 0
    covered = {}
    for path in sorted((repo / 'results/predictions').rglob('*.csv.gz')):
        columns = set(read_header(path))
        require('pair_id' in columns, f'{path.name} carries no pair_id')
        has_endpoints = {'record_id_i', 'record_id_j'} <= columns
        has_identities = {'identity_hash_i', 'identity_hash_j'} <= columns
        seen, rows = set(), 0
        for row in iter_dicts(path):
            pair = row['pair_id']
            require(pair in endpoints, f'{path.name} references an unknown pair_id {pair}')
            if has_endpoints:
                require(endpoints[pair] == (row['record_id_i'], row['record_id_j']),
                        f'{path.name} endpoint direction differs from the pair authority at {pair}')
            if has_identities:
                require((identities[row['record_id_i']], identities[row['record_id_j']])
                        == (row['identity_hash_i'], row['identity_hash_j']),
                        f'{path.name} identity keys differ from the record authority at {pair}')
            seen.add(pair)
            rows += 1
        covered[path.name] = (seen, rows)
        checked += rows
    internal_pairs, internal_rows = covered['S06_outer_predictions_long.csv.gz']
    require(len(internal_pairs) == scope['internal_unique_test_pairs'],
            'internal predictions do not cover the declared unique test pairs')
    require(internal_rows % scope['internal_test_observations'] == 0,
            'internal prediction rows are not a whole multiple of the declared test observations')
    for name, mapping in (('S20_external_ensemble_predictions.csv.gz', 'brd4_pairs.csv'),
                          ('S24_ensemble_predictions.csv.gz', 'ripk1_pairs.csv')):
        published = {row['pair_id'] for row in read_dicts(repo / 'code/external_cases' / mapping)}
        require(covered[name][0] == published, f'{name} pair set differs from {mapping}')
    return {'prediction_rows_checked': checked, 'unknown_pairs': 0,
            'endpoint_direction_mismatches': 0, 'identity_key_mismatches': 0}

def check_field_dictionary(repo):
    _, entries = payload_entries(repo)
    documented = {row['field'] for row in read_dicts(repo / 'results/field_dictionary.csv')}
    present = set()
    for entry in entries:
        if not entry['path'].endswith('.json'):
            present.update(read_header(repo / entry['path']))
    require(not present - documented, f'undocumented result columns: {sorted(present - documented)}')
    require(not documented - present, f'dictionary entries matching no column: {sorted(documented - present)}')
    return {'documented_fields': len(documented), 'distinct_result_columns': len(present)}

def check_snapshot(repo, snapshot):
    source = next(entry for entry in load_json(repo / 'data/source_manifest.json')['sources']
                  if entry.get('sha256'))
    actual = sha256_hex(snapshot)
    require(actual == source['sha256'].lower(),
            f"snapshot SHA-256 differs from {source['source_id']}: {actual}")
    return {'source_id': source['source_id'], 'sha256_matched': True,
            'bytes': snapshot.stat().st_size}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', default=Path(__file__).resolve().parent, type=Path)
    parser.add_argument('--snapshot', type=Path, help='Optional TACK DC50 parquet to checksum')
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    try:
        curation = check_curation(repo)
        report = {'status': 'RELEASE_CHECKS_PASSED',
                  'payload': check_payload(repo),
                  'curation': curation,
                  'declared_scope': check_declared_scope(repo, curation),
                  'predictions': check_predictions(repo),
                  'field_dictionary': check_field_dictionary(repo),
                  'snapshot': check_snapshot(repo, args.snapshot) if args.snapshot else 'not supplied',
                  'models_fitted': 0, 'predictions_computed': 0}
    except GateError as error:
        print(json.dumps({'status': 'RELEASE_CHECKS_FAILED', 'reason': str(error)}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0

if __name__ == '__main__':
    sys.exit(main())
