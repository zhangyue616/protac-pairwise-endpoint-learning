# Source-component isolation

This directory provides the source-component data, training-support, and prediction routes. The main analysis holds out each of 43 components with eligible pairs; component indexing also includes components without eligible pairs. The model seed is 20261017, each learner has eight candidates, and each outer cell has three inner folds. Candidate calibration and selection use the component-macro objective over 42 pair-positive training components. The shared exact calibration helper and unchanged model adapters are in `../core/`.

`scientific_components.py` preserves the original count mapping, direct antisymmetry, endpoint differences, full-training count vocabulary, and nonsingleton endpoint support. `run_loco.py` is a portable entry point for one cell. It fits all three inner layers afresh, whereas the original production run reused accepted invariant inner-model caches for some cells. This changes execution cost and storage while keeping the same scientific fits and selection rules. Training and predictions were not rerun with this entry point when packaging the repository.

Required local input directory:

- `records_revised.csv`, `pairs_revised.csv`, `fingerprints.csv`, `main_raw_sparse_counts.json`;
- `record_source_group_map.csv`, `inner_component_folds.csv`, `vocab_manifest.json`;
- vocabulary NPZ files named by the manifest, with `environment_ids` and ordered `training_record_ids`.

The current population is 699 eligible records, 511 molecular identities, and 5804 pairs. Pair targets use endpoint j minus endpoint i. Obtain third-party source materials through the documented upstream route and run the public local reconstruction entry point in `../data_build/`. The three files in `mappings/` contain only opaque record/identity/pair/component IDs and integer fold assignments; they contain no molecular structures, potency measurements, or source-reference text. Install these predefined assignments and construct full-training vocabularies in the reconstructed local input directory:

    python code/source_components/prepare_components.py --data-dir LOCAL_INPUTS

This adds the component map, inner-component folds, pair-component map and 172 count-vocabulary layers (43 components × outer/three inner layers). It does no fitting or prediction. `--component DEP_003` prepares only that component's four layers for a bounded input check. Count vocabulary construction uses every eligible training record before the endpoint singleton filter; held-out environments are discarded.

Validate one real component without fitting or prediction:

    python code/source_components/run_loco.py --data-dir LOCAL_INPUTS --output-dir LOCAL_OUTPUT --component DEP_003 --learner ridge --representation binary --pipeline endpoint_E1 --check-inputs-only

Run one cell after acquiring and preparing the required local inputs:

    python code/source_components/run_loco.py --data-dir LOCAL_INPUTS --output-dir NEW_EMPTY_OUTPUT --component DEP_003 --learner ridge --representation binary --pipeline endpoint_E1

Use a separate empty output directory per component, representation, learner, and pipeline. Learners are `ridge`, `random_forest`, `xgboost`; representations are `binary`, `count`; pipelines are `direct_pair`, `endpoint_E0`, `endpoint_E0_matched`, `endpoint_E1`. Device routing is set to CPU for Ridge/random forest and `cuda:0` for XGBoost. Every XGBoost fit must report that actual CUDA device; CPU fallback, a missing device, or another ordinal stops execution, including within each of the five E1 rounds. The CLI has no automatic queue, resume, model reuse, or external-case inference. Tree E1 performs exactly five backfitting rounds per fit unit. A completed cell writes all eight candidate OOF tables, its selection, final held-out predictions, fit receipts, and the selected model.
