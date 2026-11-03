## Fixed identity and source sensitivity analyses

These modules implement the identity cross-validation and the three source-holdout sensitivity analyses, keeping the original model, feature, candidate-grid, calibration, and prediction definitions. The main source-component analysis and its component-macro training objective are in `../source_components/` and use the same `inner_objective.py` helper supplied here.

Prepare the inputs with `../data_build/reconstruct.py`. The resulting manifest binds each private input file to its SHA-256. Inspection does not fit or import model code:

```sh
python code/core/portable.py inspect --manifest ../protac-inputs/inputs.json
```

An explicit training command runs one identity-CV unit in a fresh output directory:

```sh
python code/core/portable.py run-fold --manifest ../protac-inputs/inputs.json --representation binary --learner ridge --repeat 0 --fold 0 --pipeline direct_pair --output ../protac-runs/binary-ridge-r0-f0-direct
```

Choose `count` for the unfolded Morgan count representation. Supported pipelines are `direct_pair`, `endpoint_E0`, `endpoint_E1`, `endpoint_E0_matched`, and tree-only `context_pair_C1`. This sensitivity analysis selects and calibrates candidates using pair-pooled inner-OOF MAE. It does not substitute that objective for the main component-macro analysis.

The three source-holdout analyses use binary features and their predefined source-inner assignments:

```sh
python code/core/portable.py run-source --manifest ../protac-inputs/inputs.json --learner ridge --output ../protac-runs/source-ridge
```

The wrapper requires the scientific versions recorded in `scientific_config.json`; XGBoost training must report CUDA ordinal 0 and refuses CPU fallback or another ordinal. Before any fit, `run-fold` requires the outer identity map to be an exact five-fold cover of the eligible identities and the inner map to cover exactly the outer-training identities in three folds. It accepts only a new output directory, so checkpoint state from another input configuration cannot be silently reused. Internal engine files keep their original direct-execution guards, so invoke `portable.py`.

`run-fold` writes `INPUT_BINDING.json`, a fit ledger, and `results/{learner}_R{repeat}/outer_{fold}/{pipeline}/`, which contains candidate OOF arrays, `selection.json`, final fitted-model metadata, and `outer_predictions.csv`. Raw and calibrated predictions are stored separately. This entry point runs only the internal sensitivity analysis and does not generate the external seven-compound case-study predictions.

The final model's authoritative location is the `model_path` value in each cell's `final/unit_complete.json`; downstream external-case loading follows that pointer. Binary and count runs use different default model directories, so use this metadata rather than assuming a shared directory.
