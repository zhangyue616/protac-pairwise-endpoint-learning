"""Dormant downstream bindings: parent bridge, original RIPK1 AST, descriptive AST.

No scientific imports at module import. No original bootstrap or main is bound.
All writes are in this preparation root; original inputs and completed runs stay
read-only. Source/member selection and all statistical formulas remain original.
"""
from __future__ import annotations
import ast
import csv
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
import r5_control as control

CASE = control.BASE / 'ripk1_case'
MERGED = control.BASE / 'merged_run_output'
PROJECT = control.BASE.parents[3]


def copy_exact(source, destination):
    destination = control.target(destination)
    source = Path(source)
    expected = control.digest(source)
    if destination.exists():
        control.require(control.digest(destination) == expected, 'BRIDGE_EXISTING_BYTE_DRIFT')
        return
    temporary = destination.with_name(destination.name + '.tmp.' + str(os.getpid()))
    shutil.copyfile(source, temporary)
    control.require(control.digest(temporary) == expected, 'BRIDGE_COPY_BYTE_DRIFT')
    os.replace(temporary, destination)


def table(path):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def summarize_operation_events(rows, scope_id):
    """Pure administrative counts. Repeats use identities, never count minus420."""
    kinds = ('MODEL_LOAD', 'PREDICT_PIPELINE', 'RUNNER_PREDICT_MODEL')
    result = {kind: {'attempts': 0, 'completed': 0, 'failed': 0, 'paused_without_prediction_result': 0,
        'repeated_attempts_by_same_operation_identity': 0, 'unclosed': 0, 'unknown_events': 0} for kind in kinds}
    attempts, terminal, identities, unknown = {}, set(), {kind: set() for kind in kinds}, []
    for row in rows:
        if row.get('scope_id') != scope_id:
            continue
        kind, state = row.get('operation_kind'), row.get('operation_state')
        operation_id, identity = row.get('operation_id'), row.get('operation_identity')
        if kind not in result or state not in ('ATTEMPT', 'COMPLETE', 'FAILURE', 'PAUSED') or not operation_id or not identity:
            unknown.append(row)
            if kind in result: result[kind]['unknown_events'] += 1
            continue
        key = (kind, operation_id)
        if state == 'ATTEMPT':
            if key in attempts:
                result[kind]['unknown_events'] += 1; unknown.append(row); continue
            attempts[key] = row
            result[kind]['attempts'] += 1
            if identity in identities[kind]:
                result[kind]['repeated_attempts_by_same_operation_identity'] += 1
            identities[kind].add(identity)
        else:
            if key not in attempts or key in terminal or attempts[key]['operation_identity'] != identity:
                result[kind]['unknown_events'] += 1; unknown.append(row); continue
            terminal.add(key)
            field = {'COMPLETE': 'completed', 'FAILURE': 'failed', 'PAUSED': 'paused_without_prediction_result'}[state]
            result[kind][field] += 1
    for kind in kinds:
        result[kind]['unclosed'] = sum(key[0] == kind and key not in terminal for key in attempts)
    return {'counts': result, 'unknown_event_count': len(unknown), 'unknown_events': unknown,
        'units': {'MODEL_LOAD': 'one joblib.load call; freeze/predict scopes separate',
            'PREDICT_PIPELINE': 'one original member predict_pipeline call',
            'RUNNER_PREDICT_MODEL': 'observed runner.predict_model helper calls; member-local position0/1 identifies suboperation'},
        'nested_physical_estimator_predict_calls':'UNKNOWN_NOT_OBSERVED_BY_THIS_HOOK',
        'unclosed_or_unknown_never_inferred_complete': True}


def write_operation_sidecar(scope):
    if scope['kind'] not in ('case_freeze', 'case_predict'):
        return None
    ledger = CASE / 'MODEL_OPERATIONS.jsonl'
    rows = [json.loads(line) for line in ledger.read_text('utf-8').splitlines()] if ledger.exists() else []
    counts = summarize_operation_events(rows, scope['scope_id'])
    name = 'FREEZE_ACTUAL_OPERATION_ACCOUNTING.json' if scope['kind'] == 'case_freeze' else 'PREDICTION_ACTUAL_OPERATION_ACCOUNTING.json'
    logical = {'model_schema_member_coverage': 420, 'model_load_unique_original_members': 420} if scope['kind'] == 'case_freeze' else {
        'member_coverage': 420, 'original_member_predict_pipeline_calls': 420, 'original_model_load_unique_members': 420,
        'original_runner_predict_model_helper_suboperations_by_fixed_definition': 840}
    value = {'schema':'R5_CASE_OPERATION_ACCOUNTING_V1','scope_id':scope['scope_id'],'logical_original_protocol_coverage':logical,
        'original_fixed_receipt_counts_mean':'original unique member/pipeline coverage; not cross-resume actual attempts',
        'legacy_actual_named_freeze_fields_are_logical_coverage_under_resume':True,
        'actual_runtime_events':counts,'operation_ledger_sha256_at_snapshot':control.digest(ledger) if ledger.exists() else None,
        'operation_ledger_snapshot_rows':len(rows),'operation_ledger_hash_scope':'entire ledger prefix at this stage snapshot; later stages may append',
        'all_repeated_model_process_time_charged_to_same_budget':True,'statistical_or_scientific_values_modified':False}
    path = CASE / name
    control.atomic_json(path,value)
    return path


def parent_bridge(contract, runtime):
    """File-only relocation with explicit logical coverage; no fabricated fit events."""
    runtime.boundary()
    candidate = Path(contract['candidate_root'])
    R = candidate.parent
    queue = control.read(control.BASE / 'FIXED_REMAINING_QUEUE.json')['parent_slots']
    spec = control.read(candidate / 'spec.json')
    internal, source_members = [], []
    bindings = {}
    for index, job in enumerate(spec['jobs']):
        if index == 0:
            saved = R / 'bounded_cost_extension_2/full_protocol_batch/first_job_25'
            result = control.read(saved / 'RESULTS.json')
            control.require(result['new_fits_total_own_ledger'] == 24 and result['inherited_fits_separate'] == 1,
                'PRESERVED_INDEX0_FIT_CENSUS_DRIFT')
            original_output = saved / 'binary'
            actual_new, inherited = 24, 1
        else:
            item = next(x for x in queue if x.get('original_spec_index') == index)
            saved = control.BASE / 'parent_slots' / item['scope_id']
            result = control.read(saved / 'RESULTS.json')
            control.require(result['status'] == 'COMPLETE' and result['scope_identity_sha256'] == control.json_digest(item),
                'PARENT_BRIDGE_INCOMPLETE_OR_FOREIGN_SCOPE')
            control.require(result['artifact_index_sha256'] == control.digest(saved / 'OUTPUT_BINDINGS.json'), 'PARENT_BRIDGE_INDEX_DRIFT')
            for relative, expected in control.read(saved / 'OUTPUT_BINDINGS.json').items():
                control.require(control.digest(saved / relative) == expected, 'PARENT_BRIDGE_SOURCE_ARTIFACT_DRIFT')
            original_output = saved / job['representation']
            actual_new, inherited = result['actual_new_fits'], 0
        rep, learner, repeat, fold, pipeline = [job[k] for k in ('representation', 'learner', 'repeat', 'outer_fold', 'pipeline')]
        cell_rel = Path('results') / f'{learner}_R{repeat}' / f'outer_{fold}' / pipeline
        src_cell, cell = original_output / cell_rel, MERGED / rep / cell_rel
        selection = control.read(src_cell / 'selection.json')
        chosen = selection['selected']['candidate_id']
        for relative in ('selection.json', 'final_fit.json', 'outer_predictions.csv', 'external_predictions.csv',
                'final/unit_complete.json', f'candidates/{chosen}/candidate_complete.json', f'candidates/{chosen}/oof_predictions.npz'):
            copy_exact(src_cell / relative, cell / relative)
        summary = control.read(src_cell / 'summary.json')
        logical = 125 if pipeline == 'endpoint_E1' and learner != 'ridge' else 25
        state = {'status': 'JOB_COMPLETE', **job, 'fit_count': logical, 'fit_count_meaning': 'logical protocol coverage',
            'actual_new_fits': actual_new, 'inherited_fits': inherited, 'result': summary,
            'original_spec_index': index, 'source_result_path': (saved / 'RESULTS.json').as_posix()}
        state_path = MERGED / rep / 'runtime' / f'{learner}_R{repeat}' / f'outer_{fold}_{pipeline}.json'
        control.atomic_json(state_path, state)
        final = control.read(cell / 'final/unit_complete.json')
        model_path = Path(final['model_path'])
        entry = {'original_spec_index': index, **job, 'logical_estimator_fits': logical,
            'actual_new_fits_in_completed_parent': actual_new, 'inherited_fits': inherited,
            'model_path': model_path.as_posix(), 'model_sha256': control.digest(model_path),
            'cell_path': cell.as_posix(), 'state_path': state_path.as_posix()}
        internal.append(entry)
    s0 = control.read(R / 's0_zero_fit_branch/S0_CONTRACT.json')
    s0_complete = R / 's0_zero_fit_branch/run_output/COMPLETE_RECEIPT.json'
    s0_acceptance = R / 's0_zero_fit_branch/S0_FINAL_ACCEPTANCE.json'
    control.require(control.read(s0_complete)['corrected_members'] == 11 and control.read(s0_complete)['fits'] == 0,
        'S0_COMPLETE_BRIDGE_DRIFT')
    for member in s0['members']:
        corrected = R / 's0_zero_fit_branch/run_output/corrected_test' / member['member_id']
        receipt = control.read(corrected / 'RECEIPT.json')
        control.require(receipt['csv_sha256'] == control.digest(corrected / 'source_predictions.csv') and
            receipt['old_replay_all_eleven_passed'] is True and receipt['fits'] == 0 and
            receipt['scale_c'] == member['scale_c'], 'S0_CORRECTED_RECEIPT_DRIFT')
        cell = MERGED / 'source_holdout/results' / member['learner'] / 'C2_S0' / member['pipeline']
        copy_exact(corrected / 'source_predictions.csv', cell / 'source_predictions.csv')
        copy_exact(member['historical_selection'], cell / 'selection.json')
        source_members.append({'learner': member['learner'], 'holdout_id': 'C2_S0', 'pipeline': member['pipeline'],
            'logical_estimator_fits': 125 if member['pipeline'] == 'endpoint_E1' and member['learner'] != 'ridge' else 25,
            'actual_new_fits': 0, 'source_receipt_path': (corrected / 'RECEIPT.json').as_posix(),
            'source_receipt_sha256': control.digest(corrected / 'RECEIPT.json'), 'model_sha256': member['model_sha256'],
            'model_path': member['model_path'], 'scale_c': member['scale_c']})
    for item in (x for x in queue if x['kind'] == 'source_parent'):
        saved = control.BASE / 'parent_slots' / item['scope_id']
        result = control.read(saved / 'RESULTS.json')
        control.require(result['status'] == 'COMPLETE' and result['scope_identity_sha256'] == control.json_digest(item),
            'SOURCE_BRIDGE_INCOMPLETE_OR_FOREIGN_SCOPE')
        source_cell = saved / 'source_holdout/results' / item['learner'] / item['holdout_id'] / item['pipeline']
        cell = MERGED / 'source_holdout/results' / item['learner'] / item['holdout_id'] / item['pipeline']
        for name in ('selection.json', 'source_predictions.csv', 'summary.json'):
            copy_exact(source_cell / name, cell / name)
        source_members.append({'learner': item['learner'], 'holdout_id': item['holdout_id'], 'pipeline': item['pipeline'],
            'logical_estimator_fits': item['logical_estimator_fits'], 'actual_new_fits': result['actual_new_fits'],
            'source_result_path': (saved / 'RESULTS.json').as_posix()})
    selection = control.read(candidate / spec['source_holdout']['selected_sources'])
    # The retained source IDs/indices/seeds are unchanged. Support fields below
    # describe already saved corrected outputs, not a new source-selection rule.
    for source in selection['source_holdouts']:
        populations = []
        for learner in ('ridge', 'random_forest', 'xgboost'):
            rows = table(MERGED / 'source_holdout/results' / learner / source['holdout_id'] / 'direct_pair/source_predictions.csv')
            populations.append((len(rows), sum(r['both_identity_unseen'].lower() in ('true', '1') for r in rows)))
        control.require(len(set(populations)) == 1, 'SOURCE_CORRECTED_SAVED_SUPPORT_MISMATCH')
        source['historical_test_edges'] = source['test_edges']
        source['test_edges'], source['strict_identity_unseen_edges'] = populations[0]
    control.atomic_json(control.BASE / 'SOURCE_COMPLETION_SELECTION.json', selection)
    for learner in ('ridge', 'random_forest', 'xgboost'):
        items = [x for x in source_members if x['learner'] == learner]
        expected = 225 if learner == 'ridge' else 600
        control.require(sum(x['logical_estimator_fits'] for x in items) == expected, 'SOURCE_LOGICAL_CENSUS_DRIFT')
        control.atomic_json(MERGED / 'source_holdout/runtime' / f'{learner}_status.json', {'status': 'COMPLETE',
            'learner': learner, 'expected_fits': expected, 'logical_fit_coverage': expected,
            'actual_new_fits': sum(x['actual_new_fits'] for x in items), 'ledger_events_forged': False})
    control.require(len(internal) == 420 and len(source_members) == 33 and
        sum(x['logical_estimator_fits'] for x in internal + source_members) == 17925, 'FULL_PARENT_LOGICAL_CENSUS_DRIFT')
    for path in MERGED.rglob('*'):
        if path.is_file():
            bindings[path.as_posix()] = control.digest(path)
    manifest = {'status': 'PARENT420_SOURCE33_LOGICAL_COVERAGE_COMPLETE', 'run_id': contract['run_id'],
        'contract_sha256': runtime.contract_sha, 'internal': internal, 'source': source_members,
        'logical_estimator_fit_coverage': 17925, 'new_fits_this_remaining_queue': 17425,
        'preserved_existing_new_R5_fits': 58, 'preserved_unimported_cost_fits': 33,
        'logical_S0_inherited_fits': 475, 'S0_acceptance_path': s0_acceptance.as_posix(),
        'S0_acceptance_sha256': control.digest(s0_acceptance), 'S0_complete_sha256': control.digest(s0_complete),
        'fit_complete_events_synthesized': 0, 'artifact_bindings': bindings}
    control.atomic_json(control.BASE / 'PARENT_COMPLETION_MANIFEST.json', manifest)
    return manifest


def load_definitions(path, names, namespace):
    tree = ast.parse(Path(path).read_text('utf-8'))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    control.require({n.name for n in nodes} == set(names), 'ORIGINAL_DOWNSTREAM_AST_DEFINITION_DRIFT')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes,
        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path) + ' [original retained AST]', 'exec'), namespace)
    return namespace


class LogicalCoverage(ast.NodeTransformer):
    """Replace only administrative ledger census reads, never a scientific formula."""
    def visit_Call(self, node):
        self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == 'get' and len(node.args) == 1 and
                isinstance(node.args[0], ast.Constant) and node.args[0].value == 'FIT_COMPLETE' and
                isinstance(node.func.value, ast.Subscript) and isinstance(node.func.value.slice, ast.Constant) and
                node.func.value.slice.value == 'ledger_events'):
            return ast.Subscript(value=node.func.value.value, slice=ast.Constant('logical_fit_coverage'), ctx=ast.Load())
        if (isinstance(node.func, ast.Attribute) and node.func.attr == 'get' and len(node.args) == 1 and
                isinstance(node.args[0], ast.Constant) and node.args[0].value == 'FIT_COMPLETE' and
                isinstance(node.func.value, ast.Call) and isinstance(node.func.value.func, ast.Attribute) and
                node.func.value.func.attr == 'get' and node.func.value.args and
                isinstance(node.func.value.args[0], ast.Constant) and node.func.value.args[0].value == 'ledger_events'):
            return ast.Subscript(value=node.func.value.func.value, slice=ast.Constant('logical_fit_coverage'), ctx=ast.Load())
        return node


def adapted_case_ast(path, phase):
    tree = ast.parse(Path(path).read_text('utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == phase)
    node = LogicalCoverage().visit(node)
    if phase == 'freeze':
        boundary = next(i for i, n in enumerate(node.body) if isinstance(n, ast.Assign) and
            any(isinstance(t, ast.Name) and t.id == 'schema_receipt' for t in n.targets))
        node.body[:boundary] = ast.parse('bundle, schema, queue, spec, cfg, output, files, actual_environment = case_initial(root)').body
        model_loop = next(n for n in node.body if isinstance(n, ast.For) and ast.unparse(n.iter) == 'sorted(expected_jobs())')
        assignment = next(n for n in model_loop.body if isinstance(n, ast.Assign) and
            any(isinstance(t, ast.Name) and t.id == 'model_path' for t in n.targets))
        assignment.value = ast.parse('parent_model_path(identity)', mode='eval').body
        first = next(i for i, n in enumerate(model_loop.body) if isinstance(n, ast.Assign) and
            any(isinstance(t, ast.Name) and t.id == 'model' for t in n.targets))
        control.require(ast.unparse(model_loop.body[first + 1]) == 'model_checks.append(validate_model(model, member))' and
            isinstance(model_loop.body[first + 2], ast.Delete), 'CASE_FREEZE_LOAD_SCHEMA_AST_DRIFT')
        model_loop.body[first:first + 3] = ast.parse('model_checks.append(freeze_member(member))').body
        for n in ast.walk(node):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'copyfile' and
                    ast.unparse(n.args[0]) == "schema / 'count_unknown_environments.csv'"):
                n.args[0] = ast.Name(id='COVERAGE_SOURCE', ctx=ast.Load())
    elif phase == 'predict':
        for n in ast.walk(node):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'require' and len(n.args) == 2 and
                    isinstance(n.args[1], ast.Constant) and n.args[1].value == 'prediction already attempted; no automatic rerun'):
                n.args[0] = ast.parse('prediction_resume_allowed(manifest_path)', mode='eval').body
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and ast.unparse(n.func) == 'joblib.load':
                n.func = ast.Name(id='guarded_load', ctx=ast.Load()); n.args.append(ast.Name(id='member', ctx=ast.Load()))
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'started' for t in n.targets):
                n.value = ast.Call(func=ast.Name(id='start_declaration', ctx=ast.Load()), args=[n.value], keywords=[])
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and ast.unparse(n.func) == 'runner.predict_pipeline':
                n.func = ast.Name(id='guarded_pipeline_predict', ctx=ast.Load())
                n.args.insert(0, ast.Name(id='runner', ctx=ast.Load()))
                n.args.append(ast.Name(id='member', ctx=ast.Load()))
        member_loop = next(n for n in ast.walk(node) if isinstance(n, ast.For) and ast.unparse(n.iter) == "manifest['members']")
        member_loop.body[:0] = ast.parse('if replay_member(member, rows, checks):\n    calls += 1\n    continue').body
        member_loop.body += ast.parse('save_member(member, rows[-45:], checks[-1])').body
        catches = [n for n in ast.walk(node) if isinstance(n, ast.ExceptHandler) and n.name == 'exc']
        control.require(len(catches) == 1, 'CASE_PREDICT_CATCH_AST_DRIFT')
        catches[0].body.insert(0, ast.parse('if isinstance(exc, ControlPause):\n    raise').body[0])
    else:
        raise control.ControlError('UNDECLARED_CASE_PHASE')
    return ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])


def adapted_schema_gate_ast(path):
    tree = ast.parse(Path(path).read_text('utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'schema_gate')
    calls = [n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and
        n.func.id == 'local' and len(n.args) == 2 and
        ast.unparse(n.args[1]) == "historical.split(R5_NAME + '/', 1)[1]"]
    control.require(len(calls) == 1, 'SCHEMA_HISTORICAL_RESOLVER_AST_DRIFT')
    # The old root was R5_NAME; the new integration root is the paper project.
    # Only the historical provenance branch retains its original R5 subroot.
    calls[0].args[0] = ast.parse('root / R5_NAME', mode='eval').body
    return ast.Module(body=[node], type_ignores=[])


def bind_case(contract, runtime, phase):
    from datetime import datetime, timezone
    from importlib.metadata import version
    import itertools
    import math
    source = contract['downstream_sources']['ripk1']['path']
    namespace = dict(Path=Path, ast=ast, csv=csv, hashlib=__import__('hashlib'), importlib=__import__('importlib'),
        itertools=itertools, json=json, math=math, os=os, shutil=shutil, sys=sys, datetime=datetime, timezone=timezone,
        version=version, CASE=CASE, R5_NAME='r5_analysis', RUN_ID=contract['run_id'],
        BUNDLE_REL=Path(contract['candidate_root']).relative_to(PROJECT).as_posix(),
        SCHEMA_REL=contract['case_schema_root'], REPS=['binary', 'count'], LEARNERS=['ridge', 'random_forest', 'xgboost'],
        ARMS=['direct_pair', 'endpoint_E0', 'endpoint_E1', 'endpoint_E0_matched', 'context_pair_C1'], TOL=1e-12,
        EXPECTED_SPEC_SHA=contract['scientific_bindings'][(Path(contract['candidate_root']) / 'spec.json').as_posix()],
        EXPECTED_FREEZE_SHA=control.digest(control.BASE / 'EXECUTION_FREEZE.json'), ControlPause=control.ControlPause)
    load_definitions(source, ['GateError', 'require', 'now', 'digest', 'read_json', 'table', 'dump', 'write_table',
        'local', 'relative', 'bind', 'truth', 'combos', 'expected_jobs', 'environment_gate',
        'check_case_inputs', 'metrics', 'validate_model'], namespace)
    exec(compile(ast.fix_missing_locations(adapted_schema_gate_ast(source)), source + ' [provenance root only]', 'exec'), namespace)
    parent = control.read(control.BASE / 'PARENT_COMPLETION_MANIFEST.json')
    for name, expected in parent['artifact_bindings'].items():
        control.require(control.digest(name) == expected, 'DOWNSTREAM_PARENT_ARTIFACT_DRIFT')
    def queue_gate(bundle):
        control.require(parent['status'] == 'PARENT420_SOURCE33_LOGICAL_COVERAGE_COMPLETE' and
            len(parent['internal']) == 420 and len(parent['source']) == 33 and parent['logical_estimator_fit_coverage'] == 17925,
            'CASE_REQUIRES_COMPLETE420_AND33')
        return {'status': 'COMPLETE', 'expected_fits': 17925, 'logical_fit_coverage_not_new_fit_events': True}
    namespace['queue_gate'] = queue_gate
    original_read = namespace['read_json']
    def read_json(path):
        value = original_read(path)
        if Path(path).resolve() == Path(contract['candidate_root']) / 'scientific_config.json':
            candidate, spec = Path(contract['candidate_root']), control.read(Path(contract['candidate_root']) / 'spec.json')
            value['inputs'] = {name: {'path': str(candidate / spec['inputs'][name]), 'sha256': contract['scientific_bindings'][(candidate / spec['inputs'][name]).as_posix()]}
                for name in ('records', 'pairs', 'fingerprints')}
        return value
    namespace['read_json'] = read_json
    def case_initial(root):
        queue = queue_gate(None)
        control.require(not (CASE / 'FROZEN_MANIFEST.json').exists() and not (CASE / 'PREDICTION_STARTED.json').exists(),
            'CASE_ALREADY_FROZEN_OR_PREDICTED_NO_REFREEZE')
        candidate = Path(contract['candidate_root'])
        spec = control.read(candidate / 'spec.json')
        control.require(len(spec['jobs']) == 420 and {tuple(j[k] for k in ('representation','learner','repeat','outer_fold','pipeline'))
            for j in spec['jobs']} == namespace['expected_jobs'](), 'ORIGINAL_CASE_FIXED420_DRIFT')
        cfg = read_json(candidate / 'scientific_config.json')
        files = {}
        for path, expected in contract['scientific_bindings'].items():
            namespace['bind'](files, PROJECT, path, expected)
        namespace['bind'](files, PROJECT, control.BASE / 'PARENT_COMPLETION_MANIFEST.json')
        return candidate, PROJECT / contract['case_schema_root'], queue, spec, cfg, MERGED, files, namespace['environment_gate'](cfg)
    namespace['case_initial'] = case_initial
    def model_path(identity):
        matches = [m for m in parent['internal'] if all(m[k] == v for k, v in identity.items())]
        control.require(len(matches) == 1 and control.digest(matches[0]['model_path']) == matches[0]['model_sha256'], 'EXACT_MEMBER_MODEL_BINDING_DRIFT')
        return Path(matches[0]['model_path'])
    namespace['parent_model_path'] = model_path
    candidate = Path(contract['candidate_root'])
    namespace['COVERAGE_SOURCE'] = candidate / 'ripk1_data_preparation/count_unknown_environments.csv'
    def nearest(root, bundle, spec, schema):
        source_path = candidate / 'ripk1_data_preparation/applicability_nearest699.csv'
        copy_exact(source_path, CASE / 'applicability_nearest699.csv')
        return [{**r, 'identical_binary_feature_vector': r['identical_binary_feature_vector'].lower() in ('true','1')}
            for r in table(source_path)]
    namespace['nearest699'] = nearest
    def checkpoint_file(member, stage):
        return CASE / 'member_checkpoints' / stage / (control.json_digest(member['member_id']) + '.json')
    checkpoint_index_path = CASE / 'CHECKPOINT_BINDINGS.json'
    checkpoint_index = control.read(checkpoint_index_path) if checkpoint_index_path.exists() else {}
    for name, expected in checkpoint_index.items():
        control.require(control.digest(CASE / name) == expected, 'CASE_SAVED_MEMBER_CHECKPOINT_SHA_DRIFT')
    def save_checkpoint(path, value):
        control.atomic_json(path, value)
        checkpoint_index[path.relative_to(CASE).as_posix()] = control.digest(path)
        control.atomic_json(checkpoint_index_path, checkpoint_index)
    def check_saved(saved, member, manifest_sha=None):
        control.require(saved['member_identity_sha256'] == control.json_digest(member) and
            saved['model_sha256'] == control.digest(PROJECT / member['model_path']), 'CASE_MEMBER_CHECKPOINT_IDENTITY_DRIFT')
        if manifest_sha is not None:
            control.require(saved['manifest_sha256'] == manifest_sha, 'CASE_MEMBER_MANIFEST_DRIFT')
    def freeze_member(member):
        path = checkpoint_file(member, 'schema')
        if path.exists():
            saved = control.read(path); check_saved(saved, member); return saved['check']
        runtime.boundary()
        import joblib
        operation_id = operation('MODEL_LOAD_ATTEMPT', member)
        try:
            model = joblib.load(PROJECT / member['model_path'])
        except BaseException as exc:
            operation('MODEL_LOAD_FAILURE', member, operation_id=operation_id, reason=repr(exc)); raise
        operation('MODEL_LOAD_COMPLETE', member, operation_id=operation_id)
        check = namespace['validate_model'](model, member)
        del model
        save_checkpoint(path, {'member_identity_sha256': control.json_digest(member),
            'model_sha256': member['model_sha256'], 'check': check})
        runtime.boundary()
        return check
    namespace['freeze_member'] = freeze_member
    def prediction_resume_allowed(path):
        marker = CASE / 'PREDICTION_STARTED.json'
        if not marker.exists():
            return True
        control.require(not (CASE / 'PREDICTION_FAILURE.json').exists(), 'CASE_REAL_FAILURE_NO_BLIND_RETRY')
        control.require(control.read(marker)['manifest_sha256'] == control.digest(path), 'CASE_STARTED_MANIFEST_DRIFT')
        # Driver admits only an exact PAUSED state with new GO; worker cannot
        # silently recover an unclosed process or failed member attempt.
        control.require(runtime.invocation != marker.parent, 'CASE_RESUME_CONTEXT_INVALID')
        return True
    namespace['prediction_resume_allowed'] = prediction_resume_allowed
    def start_declaration(value):
        marker = CASE / 'PREDICTION_STARTED.json'
        if marker.exists():
            saved = control.read(marker)
            control.require(saved['manifest_sha256'] == value['manifest_sha256'], 'CASE_FIRST_DECLARATION_DRIFT')
            return saved
        return value
    namespace['start_declaration'] = start_declaration
    operations_path = CASE / 'MODEL_OPERATIONS.jsonl'
    operation_sequence = {'value':len(operations_path.read_text('utf-8').splitlines()) if operations_path.exists() else 0}
    active_member = {'value': None, 'predict_position': 0}
    def operation(event, member, *, operation_id=None, suboperation=None, **extra):
        kind, state = event.rsplit('_',1)
        identity = control.json_digest({'scope_id':runtime.scope['scope_id'],'member_id':member['member_id'],
            'operation_kind':kind,'suboperation':suboperation})
        if operation_id is None:
            operation_id = runtime.scope['scope_id'] + ':' + str(operation_sequence['value']+1)
        control.append_event(operations_path, {'event': event, 'scope_id': runtime.scope['scope_id'],
            'contract_sha256': runtime.contract_sha, 'member_id': member['member_id'],
            'member_identity_sha256': control.json_digest(member),'operation_kind':kind,'operation_state':state,
            'operation_id':operation_id,'operation_identity':identity,'suboperation':suboperation, **extra})
        operation_sequence['value'] += 1
        return operation_id
    def guarded_load(path, member):
        runtime.boundary()
        control.require(control.digest(path) == member['model_sha256'], 'CASE_PRELOAD_MODEL_SHA_DRIFT')
        import joblib
        active_member['value'] = member
        active_member['predict_position'] = 0
        operation_id = operation('MODEL_LOAD_ATTEMPT', member)
        try:
            model = joblib.load(path)
        except BaseException as exc:
            operation('MODEL_LOAD_FAILURE', member, operation_id=operation_id, reason=repr(exc)); raise
        operation('MODEL_LOAD_COMPLETE', member, operation_id=operation_id)
        return model
    namespace['guarded_load'] = guarded_load
    def guarded_pipeline_predict(runner,model,strategy,current,indices,member):
        runtime.boundary()
        operation_id=operation('PREDICT_PIPELINE_ATTEMPT',member)
        try:
            raw=runner.predict_pipeline(model,strategy,current,indices)
        except control.ControlPause:
            operation('PREDICT_PIPELINE_PAUSED',member,operation_id=operation_id); raise
        except BaseException as exc:
            operation('PREDICT_PIPELINE_FAILURE',member,operation_id=operation_id,reason=repr(exc)); raise
        operation('PREDICT_PIPELINE_COMPLETE',member,operation_id=operation_id)
        return raw
    namespace['guarded_pipeline_predict']=guarded_pipeline_predict
    def replay_member(member, rows, checks):
        path = checkpoint_file(member, 'prediction')
        if not path.exists():
            runtime.boundary(); return False
        saved = control.read(path); check_saved(saved, member, control.digest(CASE / 'FROZEN_MANIFEST.json'))
        control.require(len(saved['rows']) == 45, 'CASE_SAVED_MEMBER_POPULATION_DRIFT')
        rows.extend(saved['rows']); checks.append(saved['check']); runtime.boundary(); return True
    namespace['replay_member'] = replay_member
    def save_member(member, rows, check):
        save_checkpoint(checkpoint_file(member, 'prediction'), {'member_identity_sha256': control.json_digest(member),
            'model_sha256': member['model_sha256'], 'manifest_sha256': control.digest(CASE / 'FROZEN_MANIFEST.json'),
            'rows': rows, 'check': check})
        runtime.boundary()
    namespace['save_member'] = save_member
    original_dump = namespace['dump']
    def dump(path, value):
        path = control.target(path)
        if path.name == 'PREDICTION_STARTED.json' and path.exists():
            control.require(control.read(path)['manifest_sha256'] == value['manifest_sha256'], 'CASE_STARTED_MARKER_DRIFT')
            return
        control.atomic_json(path, value)
    namespace['dump'] = dump
    namespace['_active_member'] = active_member
    namespace['_operation'] = operation
    tree = adapted_case_ast(source, phase)
    exec(compile(ast.fix_missing_locations(tree), source + ' [R5 downstream control]', 'exec'), namespace)
    return namespace


def descriptive(contract, runtime):
    runtime.boundary()
    import numpy as np
    import pandas as pd
    source = contract['downstream_sources']['descriptive']['path']
    names = ['require','read_json','clean','arms_for','log_ratio','metrics','estimates_from_mae','read_predictions',
        'summarize_metrics','summarize_external','summarize_sources']
    namespace = dict(Path=Path, json=json, np=np, pd=pd, LEARNERS=['ridge','random_forest','xgboost'], REPS=['binary','count'],
        ARMS=['direct_pair','endpoint_E0','endpoint_E1','endpoint_E0_matched','context_pair_C1'], TOL=1e-12,
        DEPENDENCY_PATH=Path(contract['candidate_root']) / 'input_build/dependency/pair_dependency_components.csv')
    tree = ast.parse(Path(source).read_text('utf-8'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    control.require({n.name for n in nodes} == set(names), 'DESCRIPTIVE_DEFINITIONS_DRIFT')
    for node in nodes:
        LogicalCoverage().visit(node)
        if node.name == 'read_predictions':
            for n in ast.walk(node):
                if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and ast.unparse(n.func) == 'pd.read_csv' and
                        ast.unparse(n.args[0]) == "root / 'input_build/dependency/pair_dependency_components.csv'"):
                    n.args[0] = ast.Name(id='DEPENDENCY_PATH', ctx=ast.Load())
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])), source + ' [descriptive only]', 'exec'), namespace)
    control.require('bootstrap' not in namespace and 'main' not in namespace, 'BOOTSTRAP_ENTRY_PROHIBITED')
    candidate = Path(contract['candidate_root'])
    spec = control.read(candidate / 'spec.json')
    spec['output_root'] = MERGED.relative_to(PROJECT).as_posix()
    spec['inputs'] = {k: (candidate / value).relative_to(PROJECT).as_posix() for k,value in spec['inputs'].items()}
    spec['source_holdout']['selected_sources'] = (control.BASE / 'SOURCE_COMPLETION_SELECTION.json').relative_to(PROJECT).as_posix()
    pred,members,coverage,folds,selections,parameters,count = namespace['read_predictions'](PROJECT,spec)
    repeat,aggregate,primary,comparisons,gamma,estimates = namespace['summarize_metrics'](pred)
    ensemble,external_metrics = namespace['summarize_external'](members)
    source_pred,source_metrics,source_ratios,source_selected,source_count = namespace['summarize_sources'](PROJECT,spec)
    fit_count = pd.concat([count,source_count],ignore_index=True)
    control.require(int(fit_count.successful_fits.sum()) == 17925, 'SUMMARY_LOGICAL_FIT_CENSUS_DRIFT')
    fit_count['successful_fits_meaning'] = 'logical protocol coverage; not synthesized new FIT_COMPLETE'
    frequency = selections.groupby(['representation','learner','pipeline','candidate_id']).size().rename('outer_models').reset_index()
    boundary = parameters.loc[parameters.varied_numeric_grid].groupby(['representation','learner','pipeline','parameter']).agg(
        outer_models=('candidate_id','size'),at_min=('at_numeric_min','sum'),at_max=('at_numeric_max','sum')).reset_index()
    outputs = {'outer_predictions_long.csv':pred,'edge_coverage.csv':coverage,'fold_counts.csv':folds,'repeat_metrics.csv':repeat,
        'learner_metrics.csv':aggregate,'primary_repeat_ratios.csv':primary,'representation_repeat_comparisons.csv':comparisons,
        'representation_learner_comparisons.csv':gamma,'selected_candidates.csv':selections,'selection_parameter_boundaries.csv':parameters,
        'selection_candidate_frequencies.csv':frequency,'selection_boundary_summary.csv':boundary,'external_per_model_predictions.csv':members,
        'external_ensemble_predictions.csv':ensemble,'external_metrics.csv':external_metrics,'source_predictions.csv':source_pred,
        'source_metrics.csv':source_metrics,'source_descriptive_ratios.csv':source_ratios,'source_selected_candidates.csv':source_selected,
        'fit_accounting.csv':fit_count}
    out = control.BASE / 'descriptive_summary'
    hashes = {}
    for name,frame in outputs.items():
        path = control.target(out / name)
        temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
        frame.to_csv(temp,index=False,lineterminator='\n'); os.replace(temp,path); hashes[name]=control.digest(path)
    summary = {'status':'COMPLETE_R5_ORIGINAL_PROTOCOL_DESCRIPTIVE_SUMMARY','run_id':contract['run_id'],
        'aggregation':'pair weighted pool across five folds within repeat; log; equal repeats and learners',
        'estimates':namespace['clean'](estimates),'intervals':None,'new_bootstrap_CI_pvalue_generated':False,
        'original_full_statistical_reproduction_complete':False,'original_bootstrap_item_status':'NOT_AUTHORIZED_OR_EXECUTED',
        'logical_estimator_fit_coverage':17925,'new_fits_remaining_queue':17425,'preserved_existing_new_R5_fits':58,
        'unimported_cost_fits_preserved_not_deducted':33,'scientific_fits_added_by_analysis':0,
        'zero_mae_policy':'log NA; no epsilon; no silent cell deletion; retain absolute metrics/differences',
        'parent_completion_sha256':control.digest(control.BASE / 'PARENT_COMPLETION_MANIFEST.json'),
        'RIPK1_prediction_receipt_sha256':control.digest(CASE / 'RESULTS_RECEIPT.json'),'output_hashes':hashes,
        'saved_result_acceptance_pending':True,'author_release_submission_approval':False}
    control.atomic_json(out / 'SUMMARY.json',summary)
    runtime.boundary()
    return summary


def run_stage(contract, scope, runtime):
    runtime.boundary()
    CASE.mkdir(exist_ok=True)
    if scope['kind'] == 'case_freeze':
        parent_bridge(contract,runtime)
        old_case = Path(contract['downstream_sources']['ripk1']['path']).parent
        for path in (old_case / 'authority').iterdir():
            if path.is_file(): copy_exact(path, CASE / 'authority' / path.name)
        namespace = bind_case(contract,runtime,'freeze')
        result = namespace['freeze'](PROJECT)
        outputs = ['FROZEN_MANIFEST.json','FREEZE_MODEL_SCHEMA_CHECKS.csv','applicability_nearest699.csv','count_unknown_environments.csv']
    elif scope['kind'] == 'case_predict':
        namespace = bind_case(contract,runtime,'predict')
        candidate = Path(contract['candidate_root'])
        sys.path.insert(0,str(candidate / 'engine'))
        import run_binary
        import run_count
        originals = {module:module.predict_model for module in (run_binary,run_count)}
        def guarded_prediction(original):
            def call(model,X):
                runtime.boundary()
                member=namespace['_active_member']['value']
                position=namespace['_active_member']['predict_position']
                operation_id=namespace['_operation']('RUNNER_PREDICT_MODEL_ATTEMPT',member,suboperation=position)
                try:
                    prediction=original(model,X)
                except BaseException as exc:
                    namespace['_operation']('RUNNER_PREDICT_MODEL_FAILURE',member,operation_id=operation_id,
                        suboperation=position,reason=repr(exc)); raise
                namespace['_operation']('RUNNER_PREDICT_MODEL_COMPLETE',member,operation_id=operation_id,suboperation=position)
                namespace['_active_member']['predict_position']+=1
                return prediction
            return call
        for module,original in originals.items(): module.predict_model=guarded_prediction(original)
        try:
            result=namespace['predict'](PROJECT)
        finally:
            for module,original in originals.items(): module.predict_model=original
        outputs=['member_predictions.csv','ensemble_predictions.csv','ensemble_metrics.csv','contrasts.csv','model_schema_checks.csv','RESULTS_RECEIPT.json']
    elif scope['kind'] == 'describe':
        result=descriptive(contract,runtime)
        outputs=[]
    else:
        raise control.ControlError('UNDECLARED_DOWNSTREAM_STAGE')
    accounting_path=write_operation_sidecar(scope)
    if accounting_path: outputs.append(accounting_path.name)
    root=control.BASE / 'parent_slots' / scope['scope_id']
    bindings={str((CASE/name).relative_to(root.parent.parent)):control.digest(CASE/name) for name in outputs}
    # The driver verifies all outputs using paths anchored at this stage directory.
    local_bindings={os.path.relpath(control.BASE/key,root).replace('\\','/'):value for key,value in bindings.items()}
    if scope['kind']=='describe':
        for path in (control.BASE / 'descriptive_summary').iterdir():
            if path.is_file(): local_bindings[os.path.relpath(path,root).replace('\\','/')]=control.digest(path)
    control.atomic_json(root/'OUTPUT_BINDINGS.json',local_bindings)
    control.atomic_json(root/'RESULTS.json',{'status':'COMPLETE','contract_sha256':runtime.contract_sha,
        'scope_identity_sha256':control.json_digest(scope),'artifact_index_sha256':control.digest(root/'OUTPUT_BINDINGS.json'),
        'result':result,'original_full_statistical_reproduction_complete':False})
    runtime.boundary()
