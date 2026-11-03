# Fixed Ridge objective sensitivity

This posthoc analysis holds the Ridge grid and test populations fixed, then changes the inner calibration/selection objective between pair-micro and component-macro. It includes binary/count representations and endpoint E1/matched-E0 for both identity-based internal evaluation and source-component isolation. There are 60 internal and 172 source-component outer pipeline slots, each with eight cached candidates. It performs no new inner fits. Only a changed selected alpha needs a new outer fit, with a hard cap of 232.

The original production analysis replayed 1856 candidates and all 232 old winners before fitting, and this gate is preserved here. That analysis required 176 new outer fits (43 internal, 133 source-component).

Required local files are candidate OOF predictions, selections and raw held-out predictions, corrected local input/feature tables, and identity-fold assignments/count vocabularies. These caches and third-party structure tables are not distributed in this directory. They can be generated from public code after local upstream-data reconstruction: run `../core/portable.py run-fold` for the 60 Ridge binary/count × E1/matched-E0 × three-repeat × five-fold cells, and `../source_components/run_loco.py` for the corresponding 172 source-component cells. Use separate new output directories for each cell. First run `../source_components/prepare_components.py` so the input directory also has the published opaque pair/component mapping and source-component vocabularies. The code does not recover raw predictions by dividing calibrated predictions by a scale.

Index these public CLI outputs without training or prediction:

    python code/objective_sensitivity/prepare_manifests.py --data-dir LOCAL_INPUTS --internal-runs LOCAL_IDENTITY_RUNS --loco-runs LOCAL_COMPONENT_RUNS --output-dir NEW_MANIFEST_DIRECTORY

This requires all 60/172 complete cells and writes `sensitivity.json`, `internal.json` and `components.json`. Use `sensitivity.json` with the sensitivity CLI. The normalized internal reader consumes the public core output layout directly; no internal production specification or queue is required.

The top-level manifest accepts paths relative to its own directory:

    {
      "internal_manifest": "internal.json",
      "loco_manifest": "components.json"
    }

The normalized internal manifest has `data_dir` and `jobs`. Each job contains `learner`, `representation`, `pipeline`, `repeat`, `outer_fold`, and `cell_path`, pointing to the core output's `results/ridge_RN/outer_F/PIPELINE` directory. A legacy reader option also accepts explicit `candidate_root`, `queue_root`, and `component_map` paths for saved production caches. No historical machine root is built into the code.

The component cache manifest has `data_dir` pointing to the corrected source-component local inputs and `slots` containing 172 entries. Each slot requires `outer_fold`, `representation`, `pipeline`, `cell`, `test_predictions`, `old_candidate_id`, `old_scale`, and `old_scores`. `old_scores` maps each `ridge_00` through `ridge_07` to its `scale_c` and `selection_score`. A CSV slot uses `cell/oof/REPRESENTATION/PIPELINE/ridge_NN.csv`; its columns are `pair_id`, `component_id`, `true_delta`, `raw_prediction`. Held-out tables include `learner`, `representation`, `pipeline`, `selected_candidate_id` and `scale_c` as well as those prediction columns. All file locations are resolved relative to the component manifest.

The historical cache format for component `DEP_006` can be declared with `oof_format: "legacy_npz"`. It requires `legacy_pair_table` at the manifest top level, plus `cell/candidates/ridge_NN/{candidate_complete.json,oof_predictions.npz}`. NPZ indices are resolved against that older pair table, then checked against the current pair authority. This legacy table is not silently replaced with the current 5804-pair table. Fresh CSV caches can be used for every component without that legacy path.

To rescore and replay caches without fitting or new prediction:

    python code/objective_sensitivity/run_sensitivity.py --manifest LOCAL_MANIFEST.json --output-dir NEW_EMPTY_OUTPUT --score-only

Omit `--score-only` to execute changed-alpha outer fits after the replay gate and produce full summaries. New output must be empty. All objectives and representations use identical test populations; internal repeated absolute errors are averaged per pair before unique-pair/component aggregation. Native-observation and pre-frozen common-pair readouts are also included. This analysis does not identify a causal effect of source isolation, demonstrate equivalence, or provide prospective external validation.

Full summaries also write `PAIR_MEAN_ABSOLUTE_ERRORS.csv` locally, with mean absolute errors per pair, component, domain, representation, pipeline, and objective, plus observation counts. This row-level derived-error table is not included in the public result payload.
