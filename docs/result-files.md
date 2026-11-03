# Result files and field definitions

The repository contains author-generated aggregate statistics, component-resampling artifacts, and label-free model predictions for the 699-record, 5,804-pair study. [result_manifest.json](../results/result_manifest.json) gives every payload's exact columns, row count, byte size, scope, and transformation. [field_dictionary.csv](../results/field_dictionary.csv) defines each CSV column and lists the files using it.

## Reading the tables

- MAE, RMSE, absolute-loss contributions, and raw/calibrated pair predictions use pDC50 units. Direction accuracy and similarities are dimensionless. A zero prediction differs from a missing value.
- Binary and count representations use the same declared protocol. Direct, E0, E0-matched, E1, and C1 follow their manuscript definitions. E0-matched matches E1's training support.
- Internal summaries generally average repetition-level results equally. Component-macro and pair-micro evaluation use different denominators. The Ridge sensitivity tables separate the inner selection/calibration objective from the final evaluation weighting and from the test population.
- A component contrast `MAE(E1)-MAE(E0-matched)` is negative when E1 has smaller saved error. Internal `theta_match` is instead a log MAE ratio; the dictionary names that definition explicitly.
- `raw`, `calibrated`, and `projected` identify different outputs. `mean`, `min`, `max`, and `median` are saved summaries, not additional observations. `historical` columns in the input-repair comparison tables refer to the pre-repair scientific inputs; `corrected_shadow` is the corrected-input condition. The `corrected_minus_historical` comparisons also span the historical 5,788-pair and corrected 5,804-pair populations, so they do not isolate the effect of graph repair.
- The conditional intervals keep fitted models, selection, and calibrated predictions fixed. Slope draws use 2,000 component-resampling draws with seed 20261027. They omit retraining, construction, and unresolved source-family uncertainty and do not establish nominal population coverage or significance of slope differences.
- Empty group metrics keep their original missing values and zero-contribution conventions. No numbers, rounding, signs, or row order were changed in the final publication step. This does not cover the small numerical differences (at most 2.22e-16) introduced earlier when the ten corrected S29 tables were read, filtered and rewritten with default pandas parsing. Four aggregate files omit provenance columns. The omitted `source` value in `component_loco_metrics.csv` was `MACRO_METRICS.csv + LEARNER_CONTRAST_SUMMARY.csv; scenario=corrected_shadow`, a table/scenario label rather than a machine path. One group-census table also replaces two internal provenance labels with descriptive metadata; its post hoc status and the limitation that overlapping counts are not effective N remain explicit. These changes are recorded in the manifest.

## Label-free predictions

The eight ordinary `.csv.gz` files contain raw and calibrated model outputs, calibration scales, and the opaque row/member/split identifiers needed for local joins. Source endpoint measurements, true pair differences, molecular structures, similarities, source text, and pair-level losses are omitted. The primary RIPK1 flag records the predefined evaluation layer. Decompression exactly restores the selected-column CSV bytes; gzip headers use timestamp zero.

Labels for the main analysis can be reconstructed locally from the specified upstream TACK snapshot using [the data-building route](../data/README.md). The BRD4 and RIPK1 inputs additionally require lawful access to the stated sources and local curation of their measurements and full structures. Opaque identifiers facilitate alignment without supplying those source inputs. Model-output inspection needs no source labels; recomputing error, direction accuracy, and OLS slopes does.

## Payload index

| File | Rows | Bytes | Coverage |
| --- | ---: | ---: | --- |
| [current_companion/S02_layer_support_census.csv](../results/current_companion/S02_layer_support_census.csv) | 72 | 4010 | All internal and source outer/inner training support; matched masks |
| [current_companion/S06_repeat_metrics.csv](../results/current_companion/S06_repeat_metrics.csv) | 1488 | 179866 | All repetition/stratum/raw-calibrated metrics and Zero |
| [current_companion/S07_learner_metrics.csv](../results/current_companion/S07_learner_metrics.csv) | 496 | 74707 | All repetition-aggregated metrics and support, including raw and Zero |
| [current_companion/S08_primary_estimates.csv](../results/current_companion/S08_primary_estimates.csv) | 14 | 3555 | All contrasts with actual valid/invalid conditional draws |
| [current_companion/S08_primary_repeat_ratios.csv](../results/current_companion/S08_primary_repeat_ratios.csv) | 18 | 4310 | Per-repeat strategy ratios and matched/support absolute differences |
| [current_companion/S08_representation_repeat_comparisons.csv](../results/current_companion/S08_representation_repeat_comparisons.csv) | 42 | 4589 | Per-repeat representation ratios and absolute differences |
| [current_companion/S09_selected_candidates.csv](../results/current_companion/S09_selected_candidates.csv) | 420 | 100193 | Selected candidates and frozen inner scales |
| [current_companion/S09_selection_parameter_boundaries.csv](../results/current_companion/S09_selection_parameter_boundaries.csv) | 2820 | 438557 | Full selected parameters and tested numerical endpoints |
| [current_companion/S10_main_consistency_learner_mean.csv](../results/current_companion/S10_main_consistency_learner_mean.csv) | 28 | 15623 | Full internal projection/error metrics |
| [current_companion/S10_main_consistency_repeat.csv](../results/current_companion/S10_main_consistency_repeat.csv) | 84 | 39176 | Repeat-level consistency before equal-repeat averaging |
| [current_companion/S10_external_consistency.csv](../results/current_companion/S10_external_consistency.csv) | 28 | 13238 | Full exploratory ensemble projection/error metrics |
| [current_companion/S11_E1_final_training_iteration_stability.csv](../results/current_companion/S11_E1_final_training_iteration_stability.csv) | 300 | 61334 | Five rounds for every selected final tree E1 model |
| [current_companion/S11_E1_final_iteration5_summary.csv](../results/current_companion/S11_E1_final_iteration5_summary.csv) | 4 | 821 | Actual median/min/max fourth-to-fifth changes |
| [current_companion/S12_effective_input_scope_summary.csv](../results/current_companion/S12_effective_input_scope_summary.csv) | 60 | 8623 | Every actual vocabulary input scope and unknown fraction |
| [current_companion/S12_outer_vocabulary_census.csv](../results/current_companion/S12_outer_vocabulary_census.csv) | 15 | 1445 | Actual vocabulary/record/input-class support |
| [current_companion/S13_posthoc_internal_group_census.csv](../results/current_companion/S13_posthoc_internal_group_census.csv) | 57 | 12727 | Pooled, repeat and fold input-defined group support |
| [current_companion/S14_posthoc_internal_calibrated_three_group_details.csv](../results/current_companion/S14_posthoc_internal_calibrated_three_group_details.csv) | 84 | 56616 | All current calibrated model/representation/group metrics and own-group Zero |
| [current_companion/S14_posthoc_internal_group_repeatmean_range.csv](../results/current_companion/S14_posthoc_internal_group_repeatmean_range.csv) | 224 | 152117 | Raw/calibrated group metrics and repeat sensitivity ranges |
| [current_companion/S14_posthoc_internal_actual_structural_zero_contribution.csv](../results/current_companion/S14_posthoc_internal_actual_structural_zero_contribution.csv) | 6 | 804 | Actual per-repeat denominator and structural-zero contributions |
| [current_companion/S16_source_metrics.csv](../results/current_companion/S16_source_metrics.csv) | 300 | 32979 | Every current source/scenario/arm/raw-calibrated metric and Zero |
| [current_companion/S16_source_selected_candidates.csv](../results/current_companion/S16_source_selected_candidates.csv) | 33 | 7270 | Source candidate and calibration selections |
| [current_companion/S20_external_metrics.csv](../results/current_companion/S20_external_metrics.csv) | 62 | 7638 | All current exploratory raw/calibrated ensemble metrics and Zero |
| [current_companion/S21_external_member_prediction_diagnostics.csv](../results/current_companion/S21_external_member_prediction_diagnostics.csv) | 420 | 30458 | Every current member output class/zero diagnostic |
| [current_companion/S21_external_ensemble_prediction_diagnostics.csv](../results/current_companion/S21_external_ensemble_prediction_diagnostics.csv) | 28 | 2646 | Every current ensemble output class/zero diagnostic |
| [current_companion/S21_pair_input_class_census.csv](../results/current_companion/S21_pair_input_class_census.csv) | 120 | 10270 | All actual Direct d and C1 [d,s] class counts |
| [current_companion/S22_posthoc_external_calibrated_three_group_details.csv](../results/current_companion/S22_posthoc_external_calibrated_three_group_details.csv) | 84 | 23977 | All current exploratory ensemble/representation/group metrics and own-group Zero |
| [current_companion/S22_posthoc_external_group_member_repeatmean_range.csv](../results/current_companion/S22_posthoc_external_group_member_repeatmean_range.csv) | 224 | 132606 | Member loss kept separate from ensemble loss |
| [current_companion/S24_ensemble_metrics.csv](../results/current_companion/S24_ensemble_metrics.csv) | 116 | 19217 | Current RIPK1 frozen records or corrected internal-member case outputs |
| [current_companion/S25_contrasts.csv](../results/current_companion/S25_contrasts.csv) | 160 | 17742 | Current RIPK1 frozen records or corrected internal-member case outputs |
| [current_companion/S29_GROUP_AVERAGE_CONTRASTS.csv](../results/current_companion/S29_GROUP_AVERAGE_CONTRASTS.csv) | 86 | 17285 | scenario=corrected_shadow only; component-holdout results |
| [current_companion/S29_GROUP_LEARNER_CONTRASTS.csv](../results/current_companion/S29_GROUP_LEARNER_CONTRASTS.csv) | 258 | 20199 | scenario=corrected_shadow only; component-holdout results |
| [current_companion/S29_GROUP_METRICS.csv](../results/current_companion/S29_GROUP_METRICS.csv) | 1032 | 194167 | scenario=corrected_shadow only; component-holdout results |
| [current_companion/S29_LEARNER_CONTRAST_SUMMARY.csv](../results/current_companion/S29_LEARNER_CONTRAST_SUMMARY.csv) | 6 | 640 | scenario=corrected_shadow only; component-holdout results |
| [current_companion/S29_LEARNER_HISTORICAL_COMPARISON.csv](../results/current_companion/S29_LEARNER_HISTORICAL_COMPARISON.csv) | 6 | 631 | Component-holdout table; historical comparison columns explicitly labeled where present |
| [current_companion/S29_MACRO_HISTORICAL_COMPARISON.csv](../results/current_companion/S29_MACRO_HISTORICAL_COMPARISON.csv) | 24 | 8587 | Component-holdout table; historical comparison columns explicitly labeled where present |
| [current_companion/S29_MACRO_METRICS.csv](../results/current_companion/S29_MACRO_METRICS.csv) | 24 | 3540 | scenario=corrected_shadow only; component-holdout results |
| [current_companion/S29_POOLED_HISTORICAL_COMPARISON.csv](../results/current_companion/S29_POOLED_HISTORICAL_COMPARISON.csv) | 24 | 6927 | Component-holdout table; historical comparison columns explicitly labeled where present |
| [current_companion/S29_POOLED_METRICS.csv](../results/current_companion/S29_POOLED_METRICS.csv) | 24 | 3182 | scenario=corrected_shadow only; component-holdout results |
| [current_companion/S29_PRIMARY_HISTORICAL_COMPARISON.csv](../results/current_companion/S29_PRIMARY_HISTORICAL_COMPARISON.csv) | 2 | 945 | Component-holdout table; historical comparison columns explicitly labeled where present |
| [current_companion/S29_PRIMARY_SUMMARY.csv](../results/current_companion/S29_PRIMARY_SUMMARY.csv) | 2 | 345 | scenario=corrected_shadow only; component-holdout results |
| [current_companion/S29_SCALE_DIAGNOSTIC_ALL_43.csv](../results/current_companion/S29_SCALE_DIAGNOSTIC_ALL_43.csv) | 2 | 564 | scenario=corrected_shadow only; component-holdout results |
| [current_companion/S29_SENSITIVITY.csv](../results/current_companion/S29_SENSITIVITY.csv) | 90 | 10957 | scenario=corrected_shadow only; component-holdout results |
| [current_companion/S29_ZERO_BY_GROUP.csv](../results/current_companion/S29_ZERO_BY_GROUP.csv) | 43 | 2849 | scenario=corrected_shadow only; component-holdout results |
| [current_companion/S26_group_census.csv](../results/current_companion/S26_group_census.csv) | 43 | 831 | Component-holdout component/input metadata |
| [current_companion/S26_group_support.csv](../results/current_companion/S26_group_support.csv) | 43 | 5226 | Component-holdout component/input metadata |
| [current_companion/component_loco_metrics.csv](../results/current_companion/component_loco_metrics.csv) | 6 | 846 | Component-holdout learner metrics and Zero baseline; corrected inputs |
| [current_companion/component_loco_support.csv](../results/current_companion/component_loco_support.csv) | 43 | 9256 | Component counts, structural support and Zero baselines; corrected inputs |
| [current_companion/component_loco_contrasts.csv](../results/current_companion/component_loco_contrasts.csv) | 43 | 3950 | Componentwise E1-minus-E0-matched absolute and bounded-relative contrasts by representation |
| [current_companion/component_loco_summary.csv](../results/current_companion/component_loco_summary.csv) | 8 | 519 | Learner-specific component-macro contrasts and descriptive structural-support associations |
| [current_companion/component_loco_sensitivity.csv](../results/current_companion/component_loco_sensitivity.csv) | 8 | 1095 | Component-omission sensitivity ranges for contrasts and descriptive structural-support associations |
| [ridge_target_sensitivity/BASELINE_REPLAY.csv](../results/ridge_target_sensitivity/BASELINE_REPLAY.csv) | 1856 | 111067 | Post-hoc Ridge target/population/weighting sensitivity; fixed candidates and test sets |
| [ridge_target_sensitivity/CANDIDATE_SCORES.csv](../results/ridge_target_sensitivity/CANDIDATE_SCORES.csv) | 3712 | 502272 | Post-hoc Ridge target/population/weighting sensitivity; fixed candidates and test sets |
| [ridge_target_sensitivity/COMPONENT_CONTRASTS.csv](../results/ridge_target_sensitivity/COMPONENT_CONTRASTS.csv) | 952 | 103350 | Post-hoc Ridge target/population/weighting sensitivity; fixed candidates and test sets |
| [ridge_target_sensitivity/COMPONENT_METRICS.csv](../results/ridge_target_sensitivity/COMPONENT_METRICS.csv) | 1904 | 197762 | Post-hoc Ridge target/population/weighting sensitivity; fixed candidates and test sets |
| [ridge_target_sensitivity/COMPONENT_OBJECTIVE_SENSITIVITY.csv](../results/ridge_target_sensitivity/COMPONENT_OBJECTIVE_SENSITIVITY.csv) | 476 | 45983 | Post-hoc Ridge target/population/weighting sensitivity; fixed candidates and test sets |
| [ridge_target_sensitivity/MATCHED_CONTRASTS.csv](../results/ridge_target_sensitivity/MATCHED_CONTRASTS.csv) | 48 | 5658 | Post-hoc Ridge target/population/weighting sensitivity; fixed candidates and test sets |
| [ridge_target_sensitivity/METRICS.csv](../results/ridge_target_sensitivity/METRICS.csv) | 96 | 10062 | Post-hoc Ridge target/population/weighting sensitivity; fixed candidates and test sets |
| [ridge_target_sensitivity/OBJECTIVE_SENSITIVITY.csv](../results/ridge_target_sensitivity/OBJECTIVE_SENSITIVITY.csv) | 24 | 2709 | Post-hoc Ridge target/population/weighting sensitivity; fixed candidates and test sets |
| [ridge_target_sensitivity/POPULATION_COUNTS.csv](../results/ridge_target_sensitivity/POPULATION_COUNTS.csv) | 16 | 1542 | Post-hoc Ridge target/population/weighting sensitivity; fixed candidates and test sets |
| [ridge_target_sensitivity/SELECTIONS.csv](../results/ridge_target_sensitivity/SELECTIONS.csv) | 464 | 78648 | Post-hoc Ridge target/population/weighting sensitivity; fixed candidates and test sets |
| [conditional_slopes/FOUR_SLOPE_TABLE.csv](../results/conditional_slopes/FOUR_SLOPE_TABLE.csv) | 4 | 419 | Fixed-prediction conditional component-resampling slope results; not retraining uncertainty |
| [conditional_slopes/internal_binary_slope_draws.csv](../results/conditional_slopes/internal_binary_slope_draws.csv) | 2000 | 62355 | Fixed-prediction conditional component-resampling slope results; not retraining uncertainty |
| [conditional_slopes/internal_component_multiplicities.csv](../results/conditional_slopes/internal_component_multiplicities.csv) | 2000 | 161209 | Fixed-prediction conditional component-resampling slope results; not retraining uncertainty |
| [conditional_slopes/internal_count_slope_draws.csv](../results/conditional_slopes/internal_count_slope_draws.csv) | 2000 | 62319 | Fixed-prediction conditional component-resampling slope results; not retraining uncertainty |
| [conditional_slopes/loco_binary_slope_draws.csv](../results/conditional_slopes/loco_binary_slope_draws.csv) | 2000 | 63768 | Fixed-prediction conditional component-resampling slope results; not retraining uncertainty |
| [conditional_slopes/loco_component_multiplicities.csv](../results/conditional_slopes/loco_component_multiplicities.csv) | 2000 | 181249 | Fixed-prediction conditional component-resampling slope results; not retraining uncertainty |
| [conditional_slopes/loco_count_slope_draws.csv](../results/conditional_slopes/loco_count_slope_draws.csv) | 2000 | 63968 | Fixed-prediction conditional component-resampling slope results; not retraining uncertainty |
| [conditional_slopes/internal_component_order.json](../results/conditional_slopes/internal_component_order.json) | 38 | 497 | Resampling component order for saved multiplicity columns |
| [conditional_slopes/loco_component_order.json](../results/conditional_slopes/loco_component_order.json) | 43 | 562 | Resampling component order for saved multiplicity columns |
| [predictions/current_companion/S06_outer_predictions_long.csv.gz](../results/predictions/current_companion/S06_outer_predictions_long.csv.gz) | 96600 | 10776570 | Complete current outer edge predictions |
| [predictions/current_companion/S16_source_predictions.csv.gz](../results/predictions/current_companion/S16_source_predictions.csv.gz) | 28248 | 2888328 | All current edge-level source-held-out predictions |
| [predictions/current_companion/S20_external_ensemble_predictions.csv.gz](../results/predictions/current_companion/S20_external_ensemble_predictions.csv.gz) | 588 | 8842 | Current 21-edge ensembles |
| [predictions/current_companion/S20_external_per_model_predictions.csv.gz](../results/predictions/current_companion/S20_external_per_model_predictions.csv.gz) | 8820 | 123676 | All 15 current members for each model combination |
| [predictions/current_companion/S24_member_predictions.csv.gz](../results/predictions/current_companion/S24_member_predictions.csv.gz) | 18900 | 558485 | Current RIPK1 frozen records or corrected internal-member case outputs |
| [predictions/current_companion/S24_ensemble_predictions.csv.gz](../results/predictions/current_companion/S24_ensemble_predictions.csv.gz) | 1305 | 32207 | Current RIPK1 frozen records or corrected internal-member case outputs |
| [predictions/current_companion/S29_all_corrected_source_predictions.csv.gz](../results/predictions/current_companion/S29_all_corrected_source_predictions.csv.gz) | 139296 | 9297447 | Complete component-holdout raw and calibrated predictions over 5804 pairs and 43 components; source labels omitted |
| [predictions/ridge_target_sensitivity/OUTER_PREDICTIONS.csv.gz](../results/predictions/ridge_target_sensitivity/OUTER_PREDICTIONS.csv.gz) | 74032 | 4276188 | Post-hoc Ridge target/population/weighting sensitivity; fixed candidates and test sets |
