## Fixed identity and source sensitivity analyses

These modules implement the fixed identity cross-validation and three retained source-holdout sensitivity analyses. They preserve the original model, feature, candidate-grid, calibration and prediction definitions. The main source-component analysis and its component-macro training objective are in `../source_components/` and use the unchanged `inner_objective.py` helper supplied here.

Prepare the inputs with `../data_build/reconstruct.py`. The resulting manifest binds each private input file to its SHA-256. Inspection does not fit or import model code:

```sh
python code/core/portable.py inspect --manifest ../protac-inputs/inputs.json
```

An explicit training command runs one identity-CV unit in a fresh output directory:

```sh
python code/core/portable.py run-fold --manifest ../protac-inputs/inputs.json --representation binary --learner ridge --repeat 0 --fold 0 --pipeline direct_pair --output ../protac-runs/binary-ridge-r0-f0-direct
```

Choose `count` for the unfolded Morgan count representation. Supported pipelines are `direct_pair`, `endpoint_E0`, `endpoint_E1`, `endpoint_E0_matched`, and tree-only `context_pair_C1`. This sensitivity analysis selects and calibrates candidates using pair-pooled inner-OOF MAE. It does not substitute that objective for the main component-macro analysis.

The three retained source holdouts use binary features and their fixed source-inner assignments:

```sh
python code/core/portable.py run-source --manifest ../protac-inputs/inputs.json --learner ridge --output ../protac-runs/source-ridge
```

The published wrapper requires the recorded scientific versions in `scientific_config.json`; XGBoost training requires CUDA and refuses CPU fallback. It accepts only a new output directory, so checkpoint state from another input configuration cannot be silently reused. Internal engine files retain their original direct-execution guards; invoke `portable.py`.

`run-fold` writes `INPUT_BINDING.json`, a fit ledger, and `results/{learner}_R{repeat}/outer_{fold}/{pipeline}/`. That directory contains candidate OOF arrays, `selection.json`, final fitted-model metadata and `outer_predictions.csv`. Raw and calibrated predictions remain separate. This entry runs the internal sensitivity analysis only; it does not generate the external seven-compound case-study predictions.

The release preparation checked syntax, command help, input inspection, a pure calibration case and the fixed-snapshot reconstruction. It did not rerun model training or establish a fresh end-to-end model reproduction.
