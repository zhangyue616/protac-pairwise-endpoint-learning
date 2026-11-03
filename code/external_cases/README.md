# Retrospective external cases

The two cases reuse the same 420 final identity-CV models: 28 representation/learner/strategy combinations, each containing three repeats and five folds. Each representation contributes 14 combinations and 210 members. There is no additional full-data fit or case-specific training. These are the identity-CV sensitivity models; the component-holdout models from the main analysis are not used here.

`brd4_compounds.csv` and `ripk1_compounds.csv` associate source compound numbers with author record IDs and complete-graph identity hashes. The corresponding `*_pairs.csv` files specify every endpoint, direction, and layer membership. These files contain only the author correspondence and omit measurements, structures, and source artwork.

All pair predictions and locally reconstructed labels use **pDC50(j) − pDC50(i)**. The endpoints in the pair mapping are authoritative; lexical pair names or compound-number order do not define direction. In the historical private input schema, the field named `true_delta` had the opposite sign. In the released prediction/statistics code, `true_delta` means the canonical j-minus-i target after local reconstruction.

## Source curation

| Case | Source and local curation | Retained population |
| --- | --- | --- |
| BRD4 | [Source article and SI](https://doi.org/10.1038/s41467-026-70663-1); Figure 6B, supporting chemical names and reported DC50 measurements | Compounds 6–12; seven records and 21 pairs. Compounds 13/14 are controls without continuous labels and are not evaluated. |
| RIPK1 | [Author-accepted manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC13536971/) ([DOI](https://doi.org/10.1021/acs.jmedchem.6c01156)); measurements in Tables 2–4 and corresponding source drawings | Compounds 15a–15e, 18a, 18b, 3d, 5a and 5b; ten records, 42 predefined primary pairs and 45 all-pair secondary pairs. Compounds 18c/18d are excluded. |

RIPK1 measurements refer to stable nLuc-RIPK1 / Jurkat / CRBN / Nano-Glo luciferase / 24 h / DC50. The labels use the author-accepted version; complete equivalence with the version of record is unknown. Unspecified glutarimide stereochemistry must remain unspecified. The primary layer is nonchiral binary Morgan radius-2/2048 Tanimoto ≥ 0.5 − 1e-12; the secondary layer contains every within-block pair. Both layers use the same ten compounds and one source family.

Readers must obtain the source materials lawfully, transcribe and check the full molecular graphs and reported measurements locally, and respect the source terms. The repository does not distribute the source values, full manual structure transcriptions, fingerprints, model files, or training-local vocabulary payloads, and software licensing does not grant rights to those upstream materials. Source access alone does not establish that a local graph transcription matches the graph used in the study; the identity checks below provide that comparison.

## Public ID inspection

This command uses only the standard library and does not load model packages:

```sh
python code/external_cases/portable.py inspect --case BRD4
python code/external_cases/portable.py inspect --case RIPK1
```

Join the released BRD4 S20 prediction files or RIPK1 S24 files on `pair_id`; endpoint IDs also appear in the pair maps. Member identity is `(representation, learner, repeat, outer_fold, strategy)`, with BRD4's `pipeline` equivalent to `strategy`. The explicit `member_id` in `member_manifest.json` uses this identity. There are 8,820 BRD4 and 18,900 RIPK1 member predictions. BRD4 has 588 ensemble predictions; RIPK1 has 1,260 model ensemble predictions plus 45 Zero-reference rows.

## Private local input schema

For inference, create a UTF-8 CSV outside this repository:

| Field | Meaning |
| --- | --- |
| `compound_id` | Exact source number from the case compound mapping, treated as text |
| `canonical_isomeric_smiles_full` | Reader-curated complete molecular graph; no fragments or invented stereochemistry |

Supply exactly seven BRD4 or ten RIPK1 rows. The wrapper canonicalizes with the recorded RDKit version, checks each complete identity hash, computes the unchanged Morgan features, preserves the mapped pair direction, and checks RIPK1 primary membership. Count features use each member's outer-training vocabulary; unknown environments are dropped and never enlarge it. Inference does not require measured labels.

For local error calculations, create a separate UTF-8 CSV with exactly `compound_id,DC50_nM`, one row per compound. Enter the source DC50 summaries in nM; the transform is `9 − log10(DC50_nM)`. This source transcription is the reader's responsibility. The wrapper does not infer missing measurements from published predictions.

## Model cache and inference

Use the exact scientific environment in `../core/scientific_config.json`. The inference path calls the unchanged shared `run_binary.predict_pipeline`, `run_count.predict_pipeline`, `count_features`, and model adapters. It preserves antisymmetrization, difference/sum column order, endpoint subtraction, and model-specific calibration. XGBoost must use the recorded CUDA device; the wrapper refuses a CPU model or device fallback. Ridge and random forest use their recorded CPU configurations.

There are two explicit cache contracts:

1. **Original cache**: arrange trusted local fitted models under the `cache_model_path` names in `member_manifest.json`, and the 15 outer vocabularies under `outer_vocabularies[*].cache_path`. Model and vocabulary file SHA-256 values must exactly match `member_manifest.json`. These files are not supplied or downloadable from this repository.
2. **Publicly rebuilt cache**: first reconstruct the 699-record/511-identity/5,804-pair private inputs using `../data_build/reconstruct.py`, then use `../core/portable.py run-fold` for the members listed in `member_manifest.json`. For each member, set `--representation`, `--learner`, `--repeat`, `--fold`, and `--pipeline` to that member's fields; choose `--output CACHE/portable_run_directory`. For example, member `binary|ridge|R0|O0|direct_pair` uses output `CACHE/runs/binary__ridge__R0__O0__direct_pair`. Complete all 420 units once and reuse them for both cases. Count vocabularies come from the prepared input directory. The main component-holdout fits cannot be substituted.

The rebuilt contract consumes the public worker outputs: `INPUT_BINDING.json`, `results/LEARNER_Rr/outer_f/STRATEGY/selection.json`, and `final/unit_complete.json`, which points to the final local model. It checks input/configuration bindings, member assignments, the recorded selected candidate/parameters/calibration, model dimensions, and ordered vocabulary/training-ID content hashes. Rebuilt model file bytes are recorded as the actual local model identity, separately from the original manifest's serialized-model hashes. A disagreement fails closed; the wrapper does not reselect, refit, change calibration, or substitute a model.

```sh
python code/external_cases/portable.py infer --case BRD4 --compounds /private/brd4_compounds.csv --cache /private/cache --mode original --output /private/brd4_inference
python code/external_cases/portable.py infer --case RIPK1 --compounds /private/ripk1_compounds.csv --cache /private/cache --mode rebuilt --inputs /private/inputs/inputs.json --output /private/ripk1_inference
```

Outputs require a new directory. Each inference writes label-free `member_predictions.csv`; no model fitting occurs. Original-cache inference requires the exact trusted local author cache. Publicly rebuilt inference has a documented formation path, but fresh model training and its final external predictions were **not** executed when packaging this repository, and neither mode has undergone a new end-to-end reproduction here.

## Ensemble and point estimates

The BRD4 and RIPK1 formulas are in `brd4_statistics.py` and `ripk1_statistics.py`. Raw predictions are averaged across all 15 members. Calibrated predictions are averaged **after** multiplying each member by its own inner-OOF scale. There is no shared post-ensemble calibration or performance-based member selection.

```sh
python code/external_cases/portable.py summarize --case BRD4 --members results/predictions/current_companion/S20_external_per_model_predictions.csv.gz --measurements /private/brd4_measurements.csv --output /private/brd4_statistics
python code/external_cases/portable.py summarize --case RIPK1 --members results/predictions/current_companion/S24_member_predictions.csv.gz --measurements /private/ripk1_measurements.csv --output /private/ripk1_statistics
```

These commands join private source labels by compound/record/pair IDs and preserve MAE, RMSE, strict directional accuracy, half-credit treatment of zero predictions, and true-tie exclusion. Predictions with absolute value ≤ 1e-12 score zero strict or half credit in the half-zero score; nonzero truths arbitrarily close to zero fail pending an explicit direction policy. RIPK1 includes the primary42/secondary45 layers, Zero reference, and predefined contrasts. These are single-source point estimates. No pair bootstrap, confidence interval, t-test, retraining, or new statistical design is added. The source similarity metadata column is omitted from the portable aggregate output; no metric or ensemble formula is changed.

Checks during repository packaging covered source-definition AST comparisons, ID/member correspondence against the released predictions, command help/inspection, and small synthetic checks. They did not run case inference, read real source measurements for new metric calculation, fit models, or resample observations.
