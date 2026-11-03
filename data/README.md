# Source data and local reconstruction

The study uses 699 records, 511 complete molecular identities and 5,804 compatible pairs. Internal evaluation contains 3,450 saved test observations over 2,825 unique pairs; complete component holdout covers 43 components.

[source_manifest.json](source_manifest.json) records the upstream sources, specified TACK content identity, and access limits. The source table, measured activities, molecular SMILES, fingerprints, manually transcribed full structures, and publisher attachments are not mirrored here.

Obtain the specified [TACK DC50 snapshot](https://huggingface.co/datasets/ailab-bio/TACK/resolve/0ebfb3627cfeee826c3392586d46940402237b11/DC50/train-00000-of-00001.parquet) directly from its provider under the applicable terms. Its SHA-256 is `2893ce64fe1d5ad50f6bf97aa63fde9597d11e6a1b388cfe30a7cfc79bb06875`, and the retained file matches these bytes. The original acquisition timestamp, acquisition revision, and historical generating commit were not preserved and remain unknown.

The reconstruction entry point is:

```bash
python code/data_build/reconstruct.py --snapshot /path/to/TACK.parquet --output /private/local/inputs
```

It applies the inherited curation, the 24-record exclusions, source-supported graph edits, opaque row correspondence, and the predefined pair/split definitions. The author correspondence and edit instructions preserve the intended joins without distributing source measurements or full molecular structures. Follow the repository's code documentation for its environment, validation status, and remaining training requirements.

The BRD4 case is based on [Rodrigo-Brenni et al.](https://doi.org/10.1038/s41467-026-70663-1), with seven compounds and 21 pairs. The RIPK1 case uses ten compounds, 42 primary pairs, and 45 secondary pairs from the [Lu et al. author-accepted manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC13536971/) ([DOI](https://doi.org/10.1021/acs.jmedchem.6c01156)). Source drawings were manually transcribed and checked; those full transcriptions and source measurements are not supplied. The [case code and mappings](../code/external_cases/README.md) connect source compound numbers to the released pair/member predictions and document local structure/measurement schemas, model-cache preparation, inference, and ensemble calculations. Exact case reruns require lawful source access, local curation, and local fitted models. The two RIPK1 layers reuse the same ten compounds and do not constitute independent replications.

Author-generated aggregate results and compressed, label-free predictions are listed in [result-files.md](../docs/result-files.md). The prediction files include the opaque pair, endpoint, model/member, and split identifiers needed to join predictions to the local reconstruction, while omitting measured activities, true pair differences, full structures, source text, and row-level losses. Recomputing errors or fitting models requires the locally reconstructed labels and features. A reachable source URL and this repository's software license do not grant new redistribution rights to third-party data or drawings.

| Prediction set | Rows | Contents |
| --- | ---: | --- |
| Internal holdout | 96,600 | Raw/calibrated predictions for all retained model combinations and test observations |
| Three source holdouts | 28,248 | Source-split predictions, opaque joins and identity-separation annotations |
| BRD4 ensembles / members | 588 / 8,820 | Fixed 15-member case predictions and their ensemble outputs |
| RIPK1 ensembles / members | 1,305 / 18,900 | Primary-layer flag, member correspondence and case outputs |
| Complete component holdout | 139,296 | All retained combinations over 5,804 pairs and 43 components |
| Ridge target sensitivity | 74,032 | Both inner objectives and fixed internal/component-holdout predictions |

Predictions are ordinary gzip-compressed CSV files. Decompression restores the exact label-free CSV bytes; compression changes no field values. The case labels and manually transcribed full structures still require local source curation.
