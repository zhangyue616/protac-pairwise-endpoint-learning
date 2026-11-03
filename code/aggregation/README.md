# Aggregation sources and reader replay

Fifteen archived scripts contributed to 47 of the 50 tables in `results/current_companion/`; the other three (S20, S24, S25) are external-case tables. [PRODUCER_MAP.csv](PRODUCER_MAP.csv) records functions, callers and coverage. Before release, machine paths and some internal directory names in six of these scripts were mechanically replaced (25 literal substitutions, for example `/LOCAL_PROJECT_ROOT`); no other byte differs from the originals, so internal names and status text remain as written. The complete historical programs need unpublished artifacts, and retained records do not bind every script hash to its historical execution. `build_si_tables_r3.py` is a runtime-generated file; `extract_mapping.py` also computes statistics.

## What the replay regenerates

The replay starts from the released predictions. It fits no model and makes no new prediction, so it does not test training.

| Article table | Released file | Commands below | Result on Windows 11 x86-64 |
|---|---|---|---|
| Table 1 | `S07_learner_metrics.csv`, `S08_representation_repeat_comparisons.csv` | 1, 2 | Byte-identical |
| Table 2 | `S08_primary_estimates.csv` | 1, 3 | Byte-identical |
| Table 3 | `component_loco_metrics.csv` | 1, 2, 4 | Byte-identical |
| Table 4 | `results/conditional_slopes/FOUR_SLOPE_TABLE.csv` | 1, 5, 6 | Byte-identical |
| Table 5 | `S24_ensemble_metrics.csv`, `S25_contrasts.csv` | Not in this replay | See below |

Table 1 uses the S07 rows with `scale=calibrated` and `stratum=all`, plus `pipeline=zero` with `scale=raw` for Zero; its count-minus-binary column is the mean of `absolute_mae_difference` over the three repeats in the S08 file. Table 2 uses every S08 row except `log_C1_over_direct_*`; the replay also writes these 12 rows as `table2_contrasts.csv`. Tables 3 and 4 use all rows.

Table 5 needs ten RIPK1 DC50 values curated by the reader. `code/external_cases/portable.py summarize` combines them with the released member predictions and needs no structures, models or fitting. Structures are needed only for new inference or to check a transcription against the identity hashes. It writes `ensemble_metrics.csv` and `contrasts.csv`, with the same columns and row order as `S24_ensemble_metrics.csv` and `S25_contrasts.csv`; `compare_replay.py` does not check them. Table 5 uses `layer=primary42`, `output_scale=calibrated` and, in S25, `contrast=E0matched-E1`. See [external cases](../external_cases/README.md) for the command and input format.

File-name prefixes do not always match Supporting Information table numbers; for example, the `S29_*` files are not SI Table S29.

## Commands

Install [requirements.txt](../../requirements.txt), including the NumPy 1.26.4 wheel with its bundled OpenBLAS. Obtain the fixed TACK snapshot under its source terms and follow [reconstruction](../data_build/README.md) for revision `0ebfb3627cfeee826c3392586d46940402237b11`. The file downloads as `train-00000-of-00001.parquet`; save it as `../TACK_DC50.parquet`.

Run from the repository root. Every output directory must be new and outside the checkout. `-B` keeps bytecode out of the checkout. Skip the first command only if `../protac-inputs` was built by this commit; inputs from an earlier commit pass the comparison but are not byte-identical. To rebuild, delete both `../protac-inputs` and `../protac-inputs_curated_tables`.

```sh
python -B code/data_build/reconstruct.py --snapshot ../TACK_DC50.parquet --output ../protac-inputs
python -B code/aggregation/replay_main_tables.py --inputs ../protac-inputs/inputs.json --pair-strata code/aggregation/pair_strata.csv --output ../main-tables
python -B code/aggregation/replay_primary_intervals.py --inputs ../protac-inputs/inputs.json --pair-strata code/aggregation/pair_strata.csv --output ../primary-intervals
python -B code/aggregation/replay_component_tables.py --r6-results ../main-tables/r6_before_companion_filter --output ../component-tables
python -B code/fixed_prediction_diagnostics/prepare_slope_inputs.py --repo-root . --pairs ../protac-inputs/pairs_revised.csv --output-dir ../slope-inputs
python -B code/fixed_prediction_diagnostics/slope_intervals.py --input-dir ../slope-inputs --output-dir ../slope-intervals
python -B code/aggregation/compare_replay.py . ../main-tables ../primary-intervals ../component-tables ../slope-intervals
```

`r6_before_companion_filter` holds the unfiltered aggregates written by the second command. On the checked Windows machine the six replay commands took 19.6 s.

## Checking the result

`compare_replay.py` compares the 28 files in [REPLAY_COMPARISON.csv](REPLAY_COMPARISON.csv) not marked `not_run` with their released copies. It reports byte identity and the largest absolute numeric difference, and it fails on any structural or text difference or on a numeric difference above `1e-12`. Success is exit status 0 and a last line with `"failed": 0`, `"passed": 28` and `"skipped": 9`. A `COMPLETE` status from the replay scripts only means they finished.

On 2026-11-03, on Windows 11 x86-64 (Intel i7-14700KF) with the pinned packages, all 28 compared files were byte-identical to the released files; no tolerance was involved. The same commands were then run on the same computer in a new virtual environment on Ubuntu 24.04.2 LTS under WSL2 (x86-64, Linux 6.6.87.2, glibc 2.39), with uv-managed Python 3.11.15 and all eight pinned package versions. Of 28 compared files, 7 were byte-identical and 21, including all five source files of Tables 1–4, differed only within the absolute tolerance of `1e-12`; the maximum absolute difference was `4.2327252813834093e-16` and no value differed at 10 significant digits. There were 0 failures and 9 `not_run` entries. Both systems completed release verification and input inspection. No `OPENBLAS_*` variables were set in either shell. The Ubuntu differences come from the C library's `log10` and `log`, not from OpenBLAS: substituting the Windows results for those calls made all 28 files byte-identical. macOS, ARM and other x86-64 CPU families were not checked; their math libraries and BLAS kernels can also change trailing digits, which is why the comparison uses a tolerance. For scale, on Windows without the numerical settings below, the largest difference was `1.33e-15` and no value differed at 10 significant digits.

## Numerical settings

On the checked Windows machine, byte identity needs three settings. They were identified by comparing with the released bytes. The parser settings were then checked against retained author files and the software versions against environment records; the kernels below are simply ones that reproduce the released bytes.

- **pDC50.** `reconstruct.py` reads pDC50 back through the pandas default CSV parser before forming pairs, as the study's input build did. Compared with exact parsing, this changes 32 of 699 values in the last digit. On the checked Windows machine, reconstructed labels then equal the study's labels; this relies on the parser in the pinned pandas 3.0.3. On the checked Ubuntu system, glibc's `log10` changes 2 of the 699 pDC50 values and 22 of the 5,804 pair labels by `8.88e-16`.
- **Labels and predictions.** Internal rows use default-parsed labels; source-component rows use exactly parsed labels and predictions, as `summarize_r5.read_predictions` and `aggregate_full_r6.read_csv` did.
- **BLAS kernel.** The weighted dot products in the bootstrap and slope resampling go through OpenBLAS, and their trailing digits depend on the kernel selected for the CPU. `S08_primary_estimates.csv` was computed with Python 3.11.15 and NumPy 1.26.4, and the nine slope files with Python 3.12.14 and NumPy 2.3.5; kernel names were not recorded. `replay_primary_intervals.py` sets `OPENBLAS_CORETYPE=Prescott` and `slope_intervals.py` sets `Nehalem` before loading NumPy; a value set in the shell overrides them. No kernel that ran reproduces both `S08_primary_estimates.csv` and the nine slope files. The other 18 aggregation tables did not change under any kernel that ran.

## Scope

Labels retain the endpoint-j-minus-endpoint-i direction. Archived functions and complete statements supply the formulas and aggregation order. Input loading is new code with its own population and identity checks; archived checks that need unpublished run records do not run. The existing component-analyzer adaptation replaces its two 5,788-pair checks with `expected_pairs=5804`; final S29 assembly retains default pandas read/filter/write. Component reader tables consume the preceding aggregates. The archived downstream adapter changed dependency paths and `FIT_COMPLETE` ledger reads without changing formulas; these entry points do not use that adapter.

Primary intervals use the original seed 20261027, 2,000 draws, K=38 and sorted component order. Their multiplicities match the internal slope sequence exactly; the two calculations estimate different statistics. Slope resampling additionally uses K=43 for component holdout. The `slope` column of `FOUR_SLOPE_TABLE.csv` is recomputed and must also pass a check against the released value.

Public inputs are the predictions, indices, folds, component mappings, S26 census/support and `pair_strata.csv`. The S26 files are read as inputs, not recomputed. `pair_strata.csv` supplies 5,804 opaque pair IDs plus source-reviewed `time_status` and `any_approximate_label`; it contains no measurements or structures and cannot be rebuilt from TACK alone.

Outside this replay: `component_loco_support` is not connected, because its archived statements read `normalized_source_id`, which the public mapping file omits (`record_index.csv` carries the same values as `c2_source_id`); the S16 tables are not connected, because the checks in the archived `summarize_sources` need full candidate OOF scores and completion ledgers; four historical-comparison tables need unpublished historical tables and mix the 5,788-to-5,804 population change with other changes. S26 preparation and other diagnostics were not rerun. The replay covers 28 of the 69 released aggregate files.
