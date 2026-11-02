# Fixed-prediction slope diagnostics

`slope_intervals.py` reproduces the four conditional source-component resampling intervals from saved XGBoost endpoint-E1 predictions. The estimator is pooled observation-weighted free-intercept OLS, calibrated prediction = intercept + slope × true target difference. The internal input has 3450 observations in 38 pair-positive components; the source-component input has 5804 observations in 43 components. Internal repeated observations retain the multiplicity of their component.

The released prediction tables omit upstream potency labels. After local data reconstruction, attach canonical endpoint-j-minus-endpoint-i labels from `pairs_revised.csv` and use the released four-slope baseline:

    python code/fixed_prediction_diagnostics/prepare_slope_inputs.py --repo-root . --pairs LOCAL_INPUTS/pairs_revised.csv --output-dir NEW_SLOPE_INPUTS

This joins by opaque pair ID and checks endpoint direction and all panel populations. It does no fitting, prediction or resampling. It creates the local input layout:

    OLS_PARAMETERS.json
    inputs/internal_XGB_E1_predictions.csv
    inputs/loco_XGB_E1_predictions.csv

`OLS_PARAMETERS.json` has a `panels` list with `protocol`, `representation` and `slope_b`. Saved prediction CSV columns are `representation`, `true_delta`, `calibrated_prediction` and `dependency_component` (internal) or `heldout_component_id` (source-component). Input predictions remain fixed throughout this diagnostic.

    python code/fixed_prediction_diagnostics/slope_intervals.py --input-dir LOCAL_SAVED_PREDICTIONS --output-dir NEW_EMPTY_OUTPUT

Each protocol independently initializes seed 20261027 and draws 2000 whole-component resamples with replacement. Binary/count representations share component multiplicities within protocol. The interval uses the 2.5th/97.5th percentiles with NumPy's linear method. Invalid slopes are recorded; there is no redraw, variance epsilon or component exclusion. The script checks the original four point slopes and direct weighted OLS for draws 1, 1000 and 2000. Packaging retained the existing completed computation; it did not execute a new batch of resamples.

These intervals condition on fixed predictions and omit training, model selection, calibration, data construction and unresolved source-family uncertainty. They do not establish nominal population coverage or significance of differences between slopes. No model is trained, predicted or recalibrated by this script.
