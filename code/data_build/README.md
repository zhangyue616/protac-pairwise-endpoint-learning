## Reconstruct the main-analysis inputs

Obtain the TACK DC50 source file separately from its [specified repository revision](https://huggingface.co/datasets/ailab-bio/TACK/resolve/0ebfb3627cfeee826c3392586d46940402237b11/DC50/train-00000-of-00001.parquet), subject to the source's access conditions and applicable rights. Its expected SHA-256 is `2893CE64FE1D5AD50F6BF97AA63FDE9597D11E6A1B388CFE30A7CFC79BB06875`. The script verifies those bytes before reading the snapshot.

From the repository root, using the recorded Python environment:

```sh
python code/data_build/reconstruct.py --snapshot ../TACK_DC50.parquet --output ../protac-inputs
python code/core/portable.py inspect --manifest ../protac-inputs/inputs.json
```

The output directory must be new and outside this repository. It contains locally generated structures, endpoints, pairs, fingerprints and counts; keep it private unless you have independently established the rights needed to redistribute it. Reconstruction also creates a sibling `{output-name}_curated_tables` directory containing its private intermediate tables.

The repository's `curation/` files contain author-generated record/pair identifiers, source-row membership, normalized target and opaque context/source assignments, predefined split assignments, and seven previously adopted atom-edit operations. They contain no source SMILES, complete corrected structures, source measurements, or source free text; the reader's downloaded source supplies the structures and DC50 values at execution time. The graph edits preserve the downloaded source atom order, add the recorded bridge and the one recorded N-methyl group, and reproduce the recorded final stereochemical identity hashes. No new structure interpretation or split generation occurs.

The command reconstructs 699 records, 511 chemical identities, and 5,804 compatible pairs, then generates the 60 training-local count vocabularies. It checks every final identity and the complete compatible-edge population against the author index, preserving pair direction and physical-member assignments. The resulting `inputs.json` binds the output files by SHA-256 for the model entry points.

The main output files are `records_revised.csv`, `pairs_revised.csv`, `fingerprints.csv`, `main_raw_sparse_counts.json`, the predefined identity-fold tables, source-holdout metadata, and `vocab/`. Additional source-component assignments and vocabularies for the main analysis are prepared by `code/source_components/prepare_components.py`; see that directory's instructions.

In the reconstructed input `pairs_revised.csv`, `canonical_target_b_minus_a` is the signed target pDC50(j) − pDC50(i), while the historical compatibility column `true_delta` stores its negative. Model entry points check this input sign convention and use the canonical target; their prediction/result output column `true_delta` contains the forward target pDC50(j) − pDC50(i).

During repository packaging, these inputs were reconstructed from the specified snapshot in a new private directory. The 699 record scientific fields and fingerprints matched the original analysis inputs, as did all sparse counts and all 60 vocabulary/member arrays. Pair identifiers, endpoints, directions, and labels also matched, with numeric CSV differences of at most `3.61e-15` (acceptance tolerance `1e-12`). This check covers model-consumed main-analysis fields and excludes ancillary source-audit text columns. It does not reconstruct the historical upstream databases-to-TACK transformation, recertify source chemistry, redistribute the source dataset, or rerun model training.

`prepare_inputs.py` is also available for holders of the original final curated tables. Its direct CLI checks their recorded file identities before preparing counts and vocabularies. The raw-snapshot route above is the public reconstruction entry point.
