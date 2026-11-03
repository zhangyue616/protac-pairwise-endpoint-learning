"""Bind the existing R5 diagnostic functions to the completed corrected run.

No estimator import/load/predict/fit. Only the final E1 logs are needed for S11;
they remain at the parent paths recorded by the completed final-fit receipts.
"""
from pathlib import Path
import hashlib
import json
import sys
from datetime import datetime, timezone

sys.dont_write_bytecode = True
SOURCE = Path('/LOCAL_PROJECT_ROOT/r5_analysis/analysis/diagnostics')
sys.path.insert(0, str(SOURCE))
import derive_r5_diagnostics as original
import diagnostic_math
import numpy as np
import pandas as pd

# Provenance labels only; the imported scientific formulas remain unchanged.
original.POSTHOC = diagnostic_math.POSTHOC = 'post_hoc_R5_corrected_input_diagnostic_update_20261016'

OUT = Path(__file__).resolve().parent
REPAIR = OUT.parents[1]
RUN = REPAIR / 'r5_remaining_20261013'
CONTRACT = original.read_json(RUN / 'EXECUTION_CONTRACT.json')
BUNDLE = Path(CONTRACT['candidate_root'])
SPEC = original.read_json(BUNDLE / 'spec.json')
SUMMARY = RUN / 'descriptive_summary'


def final_stability(jobs):
    rows, bindings = [], []
    for job in jobs:
        if job['pipeline'] != 'endpoint_E1' or job['learner'] == 'ridge':
            continue
        cell = RUN / 'merged_run_output' / job['representation'] / 'results' / f"{job['learner']}_R{job['repeat']}" / f"outer_{job['outer_fold']}" / 'endpoint_E1'
        receipt_path = cell / 'final_fit.json'
        receipt = original.read_json(receipt_path)
        log_path = Path(receipt['iteration_diagnostics'])
        original.require(log_path.is_relative_to(RUN / 'parent_slots'), 'final log outside corrected parent slots')
        value = original.read_json(log_path)
        original.require(value['training_only'] is True and value['fixed_iterations'] == 5, 'final E1 diagnostic provenance differs')
        original.require([r['iteration'] for r in value['rounds']] == [1, 2, 3, 4, 5], 'final E1 rounds missing')
        original.require(receipt['fit_count'] == 5 and len(receipt['fit_receipts']) == 5, 'final E1 receipt rounds missing')
        for saved, fitted in zip(value['rounds'], receipt['fit_receipts']):
            original.require(all(fitted[k] == v for k, v in saved.items()), 'final log/fit receipt round drift')
        original.require(value['final_h_shift'] == receipt['final_h_shift'], 'final h shift drift')
        selected = original.read_json(cell / 'selection.json')['selected']['candidate_id']
        rows.extend({**job, 'phase': 'outer_final', 'candidate_id': selected, 'inner_fold': -1,
                     'diagnostic_path': log_path.as_posix(), 'final_h_shift': value['final_h_shift'], **r}
                    for r in value['rounds'])
        bindings.append({**job, 'final_fit_path': receipt_path.as_posix(),
                         'diagnostic_path': log_path.as_posix(), 'model_path_reference_only': receipt['model_path']})
    table = pd.DataFrame(rows).sort_values(['representation', 'learner', 'repeat', 'outer_fold', 'iteration'])
    original.require(len(bindings) == 60 and len(table) == 300, 'tree final 60x5 closure fails')
    fifth = table.loc[table.iteration.eq(5)]
    metrics = ['training_h_change_rmse', 'alpha_change_record_weighted_rmse']
    compact = fifth.groupby(['representation', 'learner'])[metrics].agg(['median', 'min', 'max'])
    compact.columns = [f'{metric}_{stat}' for metric, stat in compact.columns]
    compact = compact.reset_index()
    compact['final_models'] = fifth.groupby(['representation', 'learner']).size().to_numpy()
    return {'E1_final_training_iteration_stability.csv': table,
            'E1_final_iteration5_summary.csv': compact,
            'E1_final_training_log_bindings.csv': pd.DataFrame(bindings)}


def main():
    original.require(OUT.name == 'diagnostics' and OUT.parent.name == 'r5_postprocessing_20261016', 'output boundary differs')
    original.require(CONTRACT['status'] == 'FROZEN_EXECUTION_CONTRACT', 'corrected contract not frozen')
    config = original.read_json(BUNDLE / SPEC['scientific_config'])
    main_population = original.load_population(BUNDLE, SPEC['inputs'])
    external = original.load_population(BUNDLE, SPEC['external'])
    outer = pd.read_csv(BUNDLE / SPEC['inputs']['outer_folds'])
    outer_maps = {int(r): dict(zip(cell.identity_hash, cell.outer_fold)) for r, cell in outer.groupby('repeat')}
    for om in outer_maps.values():
        original.require(set(om) == set(main_population.records.identity_hash), 'corrected identity fold scope differs')
    # The stage numerical owner accepts the 420 member closure. Here we bind the
    # saved long tables and verify the population joins used by the diagnostics.
    pred = pd.read_csv(SUMMARY / 'outer_predictions_long.csv').drop(columns=['dependency_component'])
    members = pd.read_csv(SUMMARY / 'external_per_model_predictions.csv')
    for frame, population in [(pred, main_population), (members, external)]:
        truth = population.pairs.set_index('pair_id').true_delta.reindex(frame.pair_id).to_numpy(float)
        original.require(np.isfinite(truth).all() and np.allclose(truth, frame.true_delta, rtol=0, atol=1e-12), 'corrected prediction/population truth differs')
    original.require(len(pred) == 3450 * 28 and len(members) == 420 * 21, 'corrected long prediction coverage differs')
    ensemble = original.external_ensemble(members)
    outputs = {'main_outer_predictions.csv': pred, 'external_per_model_predictions.csv': members,
               'external_ensemble_predictions.csv': ensemble}
    print('Bound corrected populations and saved predictions', flush=True)
    outputs.update(original.consistency_tables(pred, 'main'))
    outputs.update(original.consistency_tables(ensemble, 'external'))
    potentials = outputs['external_projected_potentials.csv']
    observed = dict(zip(external.records.record_id, external.z - external.z.mean()))
    potentials['observed_centered_pdc50'] = potentials.record_id.map(observed)
    potentials['potential_note'] = 'projected batch edge predictions; no per-entity error bars inferred'
    outputs.update(final_stability(SPEC['jobs']))
    print('Completed graph projection and final E1 log recovery', flush=True)
    representation, binary_maps, count_maps, emaps = original.representation_tables(
        BUNDLE, SPEC, main_population, external, outer_maps, members, ensemble)
    outputs.update(representation)
    outputs['external_member_prediction_diagnostics.csv'] = original.output_value_diagnostics(members)
    outputs['external_ensemble_prediction_diagnostics.csv'] = original.output_value_diagnostics(ensemble)
    print('Completed corrected feature/vocabulary class mapping', flush=True)
    outputs.update(original.posthoc_tables(BUNDLE, SPEC, config, pred, members, ensemble,
                                          main_population, external, binary_maps, count_maps, emaps, OUT))
    manifest = []
    for name, frame in outputs.items():
        original.write_csv(OUT / name, frame)
        manifest.append({'file': name, 'rows': len(frame), 'columns': list(frame.columns)})
    bound = [Path(__file__), SOURCE / 'derive_r5_diagnostics.py', SOURCE / 'diagnostic_math.py',
             RUN / 'EXECUTION_CONTRACT.json', BUNDLE / 'spec.json',
             SUMMARY / 'outer_predictions_long.csv', SUMMARY / 'external_per_model_predictions.csv',
             BUNDLE / config['estimands']['conditional_bootstrap']['components_source']]
    receipt = {'status': 'COMPLETE_CORRECTED_R5_DERIVED_DIAGNOSTICS_PENDING_STAGE_ACCEPTANCE',
               'completed_utc': datetime.now(timezone.utc).isoformat(), 'run_id': CONTRACT['run_id'],
               'candidate_root': BUNDLE.as_posix(), 'summary_root': SUMMARY.as_posix(),
               'original_code': SOURCE.as_posix(),
               'binding_adaptation': 'completed corrected contract replaces legacy frozen bundle gate; long saved predictions replace repeated member reads; final E1 log paths taken from corrected final_fit receipts',
               'scientific_formulas_changed': False, 'fits': 0, 'models_loaded': 0, 'new_predictions': 0,
               'main_records': len(main_population.records), 'main_identities': int(main_population.records.identity_hash.nunique()),
               'main_pairs': len(main_population.pairs), 'internal_test_observations': 3450,
               'external_records': len(external.records), 'external_pairs': len(external.pairs),
               'final_tree_E1_models': 60, 'final_tree_E1_rounds': 300,
               'scope': 'existing input equality, zero inputs/outputs, three-group support/loss, graph projection, final E1 round stability',
               'omitted_not_needed': 'inner candidate iteration logs are not recopied; S11 uses all 60 selected outer final models',
               'missing_requested_diagnostics': [],
               'bound_sources': [{'path': p.as_posix(), 'sha256': original.sha256(p)} for p in bound],
               'files': manifest}
    original.dump(OUT / 'DIAGNOSTICS_RECEIPT.json', receipt)
    print(json.dumps({k: receipt[k] for k in ['status', 'main_records', 'main_identities', 'main_pairs', 'fits', 'models_loaded', 'new_predictions']}), flush=True)


if __name__ == '__main__':
    main()
