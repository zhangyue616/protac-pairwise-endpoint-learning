"""Compute outcome-blind structural support; never import a model or read results.

The input group map must bind every record to exactly one dependency_component.
This is metadata preparation, not a frozen execution protocol or fit permission.
"""
from pathlib import Path
import argparse
import csv
import json
from collections import Counter
from statistics import mean, median


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        raise RuntimeError("No rows to write: " + str(path))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = read_csv(args.inputs / "records_revised.csv")
    pairs = read_csv(args.inputs / "pairs_revised.csv")
    fingerprints = read_csv(args.inputs / "fingerprints.csv")
    groups = read_csv(args.groups)
    record_by_id = {row["record_id"]: row for row in records}
    if len(record_by_id) != len(records):
        raise RuntimeError("Duplicate record IDs")
    group_by_id = {row["record_id"]: row["dependency_component"] for row in groups}
    if len(group_by_id) != len(groups) or set(group_by_id) != set(record_by_id):
        raise RuntimeError("Group map is not an exact unique record map")
    bits_by_identity = {}
    for row in fingerprints:
        identity = record_by_id[row["record_id"]]["identity_hash"]
        bits = int(row["fp_hex"], 16)
        if identity in bits_by_identity and bits_by_identity[identity] != bits:
            raise RuntimeError("Fingerprint differs within identity")
        bits_by_identity[identity] = bits
    pairs_by_group = {}
    for row in pairs:
        group_i = group_by_id[row["record_id_i"]]
        if group_i != group_by_id[row["record_id_j"]]:
            raise RuntimeError("Pair crosses source/identity components")
        pairs_by_group.setdefault(group_i, []).append(row)
    endpoint_rows, pair_rows, group_rows = [], [], []
    for group in sorted(pairs_by_group):
        train = [r for r in records if group_by_id[r["record_id"]] != group]
        blocks = Counter((r["normalized_poi_id"], r["pair_context_hash"]) for r in train)
        matched = [r for r in train if blocks[r["normalized_poi_id"], r["pair_context_hash"]] >= 2]
        train_ids = sorted({r["identity_hash"] for r in matched})
        test_pairs = pairs_by_group[group]
        test_ids = sorted({p[key] for p in test_pairs for key in ("identity_hash_i", "identity_hash_j")})
        if not train_ids or set(train_ids) & set(test_ids):
            raise RuntimeError("Empty support or identity overlap")
        support = {}
        for identity in test_ids:
            a = bits_by_identity[identity]
            scores = []
            for train_id in train_ids:
                b = bits_by_identity[train_id]
                union = (a | b).bit_count()
                if union == 0:
                    raise RuntimeError("Undefined all-zero fingerprint similarity")
                scores.append(((a & b).bit_count() / union, train_id))
            maximum = max(s for s, _ in scores)
            nearest = min(i for s, i in scores if s == maximum)
            support[identity] = maximum
            endpoint_rows.append({"dependency_component": group, "test_identity_hash": identity,
                                  "matched_train_identities": len(train_ids),
                                  "nearest_train_identity_hash": nearest,
                                  "max_binary_tanimoto": maximum})
        for pair in test_pairs:
            a, b = support[pair["identity_hash_i"]], support[pair["identity_hash_j"]]
            pair_rows.append({"dependency_component": group, "pair_id": pair["pair_id"],
                              "endpoint_i_max_tanimoto": a, "endpoint_j_max_tanimoto": b,
                              "pair_min_support": min(a, b)})
        group_rows.append({"dependency_component": group, "test_pairs": len(test_pairs),
                           "test_pair_endpoint_identities": len(test_ids),
                           "matched_train_records": len(matched), "matched_train_identities": len(train_ids),
                           "primary_support_median_identity_nn": median(support.values()),
                           "support_mean_identity_nn": mean(support.values()),
                           "support_min_identity_nn": min(support.values()),
                           "support_max_identity_nn": max(support.values()),
                           "secondary_pair_mean_min_support": mean(min(support[p["identity_hash_i"]], support[p["identity_hash_j"]]) for p in test_pairs)})
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "endpoint_support.csv", endpoint_rows)
    write_csv(args.output / "pair_support.csv", pair_rows)
    write_csv(args.output / "group_support.csv", group_rows)
    receipt = {"status": "OUTCOME_BLIND_METADATA_PREPARATION_ONLY", "group_map": str(args.groups.resolve()),
               "input_directory": str(args.inputs.resolve()), "groups_with_test_pairs": len(group_rows),
               "primary_support": "median over distinct held-out pair-endpoint identities of max binary Morgan Tanimoto to outer-train E1/E0-matched-supported identities",
               "tie_rule": "lexicographically first nearest identity; score unchanged",
               "models_imported": False, "fits": 0, "predictions": 0, "new_test_errors_read": False,
               "test_labels_used": False, "selection_by_support_or_error": False,
               "scope": "Candidate source/shared-identity map; no family-independent validation claim. Final map changes require this preparation to be regenerated before protocol freeze."}
    (args.output / "SUPPORT_PREPARATION.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"groups": len(group_rows), "pair_rows": len(pair_rows), "fits": 0, "new_test_errors_read": False}))


if __name__ == "__main__":
    main()
