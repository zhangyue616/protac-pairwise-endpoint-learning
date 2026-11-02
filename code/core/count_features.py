"""Unfolded Morgan environment counts with strictly training-local vocabularies."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator


def generator():
    # Atom invariants are the generator's standard connectivity invariants,
    # rather than feature invariants. fpSize affects GetFingerprint only.
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False,
        useBondTypes=True, onlyNonzeroInvariants=False,
        includeRingMembership=True, includeRedundantEnvironments=False)


def signature(counts: dict[int, int]) -> str:
    payload = json.dumps(sorted(counts.items()), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest().upper()


def compute_counts(records: pd.DataFrame, fp_table: pd.DataFrame):
    gen = generator()
    binary = fp_table.set_index("record_id").fp_hex.to_dict()
    rows, counts = [], []
    for row in records.itertuples(index=False):
        mol = Chem.MolFromSmiles(row.canonical_isomeric_smiles_full)
        if mol is None:
            raise RuntimeError(f"structure parsing failed: {row.record_id}")
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        identity = hashlib.sha256(canonical.encode()).hexdigest().upper()
        if identity != row.identity_hash:
            raise RuntimeError(f"structure identity mismatch: {row.record_id}")
        count = {int(k): int(v) for k, v in gen.GetSparseCountFingerprint(mol).GetNonzeroElements().items()}
        bits = set(gen.GetFingerprint(mol).GetOnBits())
        if {k % 2048 for k in count} != bits:
            raise RuntimeError("sparse environment namespace differs from binary namespace")
        recomputed = format(sum(1 << b for b in bits), "0512X")
        if recomputed != binary[row.record_id]:
            raise RuntimeError(f"binary-cache mismatch: {row.record_id}")
        counts.append(count)
        rows.append({"record_id": row.record_id, "identity_hash": row.identity_hash,
            "binary_2048_signature": hashlib.sha256(recomputed.encode()).hexdigest().upper(),
            "raw_sparse_count_signature": signature(count), "distinct_environment_ids": len(count),
            "total_environment_count": sum(count.values()), "binary_cache_match": True})
    return counts, pd.DataFrame(rows)


class PairFeatures:
    """Only materialize the selected pair rows, avoiding full-graph dense copies."""
    def __init__(self, X: np.ndarray, pi: np.ndarray, pj: np.ndarray, operation: str):
        self.X, self.pi, self.pj, self.operation = X, pi, pj, operation

    def __getitem__(self, indices):
        a, b = self.X[self.pi[indices]], self.X[self.pj[indices]]
        return b - a if self.operation == "difference" else b + a


def map_counts(counts: list[dict[int, int]], vocab: np.ndarray):
    column = {int(k): p for p, k in enumerate(vocab)}
    X = np.zeros((len(counts), len(vocab)), np.float32)
    unknown = []
    for i, count in enumerate(counts):
        lost_ids, lost_counts = 0, 0
        for k, value in count.items():
            pos = column.get(k)
            if pos is None:
                lost_ids += 1
                lost_counts += value
            else:
                X[i, pos] = value
        unknown.append({"unknown_environment_ids": lost_ids, "raw_environment_ids": len(count),
                        "unknown_environment_count": lost_counts, "raw_environment_count": sum(count.values()),
                        "unknown_id_fraction": lost_ids / len(count) if count else 0.0,
                        "unknown_count_fraction": lost_counts / sum(count.values()) if count else 0.0})
    return X, pd.DataFrame(unknown)


def layer_name(repeat: int, outer_fold: int, inner_fold: int) -> str:
    return f"R{repeat}_O{outer_fold}_I{inner_fold}"


def class_table(records: pd.DataFrame, X: np.ndarray, unknown: pd.DataFrame,
                *, layer: str, population: str, train: np.ndarray | None = None):
    signatures = [hashlib.sha256(x.tobytes()).hexdigest().upper() for x in X]
    result = pd.DataFrame({"layer": layer, "population": population,
        "record_id": records.record_id.to_numpy(), "identity_hash": records.identity_hash.to_numpy(),
        "effective_count_signature": signatures, "effective_zero_vector": np.all(X == 0, axis=1),
        "vocabulary_size": X.shape[1]})
    if train is not None:
        result["in_vocabulary_training_scope"] = train
    if "compound_id" in records:
        result["compound_id"] = records.compound_id.to_numpy()
    return pd.concat([result, unknown.reset_index(drop=True)], axis=1)


def classes_summary(table: pd.DataFrame, field: str):
    groups = table.groupby(field, sort=True)
    distinct = groups.identity_hash.nunique()
    return {"records": len(table), "chemical_identities": table.identity_hash.nunique(),
            "representation_classes": groups.ngroups,
            "classes_with_distinct_chemical_identities": int((distinct > 1).sum()),
            "records_in_cross_identity_classes": int(groups.size()[distinct > 1].sum())}


def mapped_bundle(bundle, external, record_train: np.ndarray, context: dict[str, Any], root: Path):
    key = layer_name(int(context["repeat"]), int(context["outer_fold"]), int(context["inner_fold"]))
    path = (VOCAB_ROOT if VOCAB_ROOT is not None else root / "data/vocab") / f"{key}.npz"
    with np.load(path, allow_pickle=False) as values:
        vocab = values["environment_ids"]
        training_ids = values["training_record_ids"]
    actual_ids = bundle.records.loc[record_train, "record_id"].to_numpy(str)
    if not np.array_equal(training_ids, actual_ids):
        raise RuntimeError(f"training-local vocabulary support mismatch: {key}")
    X, _ = map_counts(bundle.raw_counts, vocab)
    mapped = replace(bundle, X_record=X,
                     X_diff=PairFeatures(X, bundle.pair_i, bundle.pair_j, "difference"),
                     X_sum=PairFeatures(X, bundle.pair_i, bundle.pair_j, "sum"))
    mapped_external = None
    if external is not None:
        E, _ = map_counts(external.raw_counts, vocab)
        mapped_external = replace(external, X_record=E,
            X_diff=PairFeatures(E, external.pair_i, external.pair_j, "difference"),
            X_sum=PairFeatures(E, external.pair_i, external.pair_j, "sum"))
    return mapped, mapped_external, {"representation": "unfolded_raw_sparse_count",
        "vocabulary_layer": key, "vocabulary_path": path.as_posix(), "vocabulary_size": len(vocab),
        "vocabulary_training_records": len(training_ids), "vocabulary_training_scope_verified": True,
        "normalization": "none", "unknown_environment_rule": "drop; no heldout vocabulary extension"}

# Bound by the portable entry only after its freeze contract passes.
VOCAB_ROOT = None
