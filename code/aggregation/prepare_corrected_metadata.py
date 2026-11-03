"""Rebuild the corrected analysis metadata with the original R6 definitions.

May run before prediction completion because it uses no outcome error values.
No estimator import, fit, load, predict, test-error lookup or support selection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def prepare(input_root, output, plan, read_csv, read_bytes, bindings):
    input_root, output = Path(input_root), Path(output)
    require(not output.exists(), "Corrected metadata requires a fresh output directory")
    records = read_csv(input_root / "records_revised.csv")
    pairs = read_csv(input_root / "pairs_revised.csv")
    mapping = read_csv(input_root / "record_source_group_map.csv")
    fingerprints = read_csv(input_root / "fingerprints.csv")
    require(len(records) == 699 and records.record_id.is_unique, "Corrected record population drift")
    require(len(pairs) == plan["expected_new_pairs"] and pairs.pair_id.is_unique,
            "Corrected pair population drift")
    require(mapping.record_id.is_unique and set(mapping.record_id) == set(records.record_id),
            "Corrected component map does not uniquely cover records")
    require(fingerprints.record_id.is_unique and set(fingerprints.record_id) == set(records.record_id),
            "Corrected fingerprints do not uniquely cover records")
    mapped = mapping.set_index("record_id").loc[records.record_id]
    require(np.array_equal(records.identity_hash.to_numpy(), mapped.identity_hash.to_numpy()),
            "Corrected record/map full identity differs")
    groups = pairs.record_id_i.map(mapping.set_index("record_id").component_id)
    require(groups.equals(pairs.record_id_j.map(mapping.set_index("record_id").component_id)),
            "Corrected pair crosses components")
    require(not groups.isna().any() and groups.nunique() == plan["expected_components"],
            "Corrected positive-pair group population drift")

    # These are exactly the five census fields consumed by the frozen analyzer;
    # their definitions follow source_inventory/build_source_inventory.py.
    record_groups = records.copy()
    record_groups["component_id"] = mapped.component_id.to_numpy()
    census_rows = []
    for component in sorted(set(groups)):
        test = record_groups.loc[record_groups.component_id.eq(component)]
        test_pairs = pairs.loc[groups.eq(component)]
        endpoint_ids = set(test_pairs.identity_hash_i) | set(test_pairs.identity_hash_j)
        census_rows.append({"component_id": component, "records": len(test),
                            "identities": test.identity_hash.nunique(), "pairs": len(test_pairs),
                            "pair_endpoint_identities": len(endpoint_ids)})
    import pandas as pd
    census = pd.DataFrame(census_rows)
    require(tuple(census.sort_values(["pairs", "component_id"], ascending=[False, True])
                  .component_id.iloc[:2]) == ("DEP_006", "DEP_002"),
            "Original prespecified largest components changed")
    require(int(census.pair_endpoint_identities.ge(5).sum()) == 25,
            "Original prespecified >=5-endpoint sensitivity population changed")
    pair_map = pairs[["pair_id", "record_id_i", "record_id_j"]].copy()
    pair_map["heldout_component_id"] = groups
    support_map = mapping[["record_id", "identity_hash", "component_id"]].rename(
        columns={"component_id": "dependency_component"})

    # The original support source is executed unchanged in memory, so only the
    # component-column interface is adapted by the separate GROUP_MAP.csv.
    support_source = read_bytes(plan["structural_support_definition_reference"])
    output.mkdir(parents=True)
    census.to_csv(output / "group_census.csv", index=False, lineterminator="\n")
    pair_map.to_csv(output / "pair_source_group_map.csv", index=False, lineterminator="\n")
    support_map.to_csv(output / "GROUP_MAP.csv", index=False, lineterminator="\n")
    namespace = {"__name__": "r6_original_support_definition",
                 "__file__": plan["structural_support_definition_reference"]}
    exec(compile(support_source, plan["structural_support_definition_reference"], "exec"), namespace)
    old_argv = sys.argv
    try:
        sys.argv = [plan["structural_support_definition_reference"],
                    "--inputs", str(input_root), "--groups", str(output / "GROUP_MAP.csv"),
                    "--output", str(output)]
        namespace["main"]()
    finally:
        sys.argv = old_argv
    support = read_csv(output / "group_support.csv")
    require(set(support.dependency_component) == set(census.component_id),
            "Corrected support does not cover exactly the selected groups")
    require((support.test_pairs.to_numpy() ==
             census.set_index("component_id").loc[support.dependency_component].pairs.to_numpy()).all(),
            "Corrected support/census pair counts differ")
    binding = {"status": "CORRECTED_METADATA_PREPARED_USING_ORIGINAL_DEFINITIONS",
               "input_root": input_root.resolve().as_posix(),
               "structural_support_definition_reference": plan["structural_support_definition_reference"],
               "census_definition_reference": plan["census_definition_reference"],
               "census_columns": list(census.columns),
               "component_column_adaptation": "component_id -> dependency_component for original support CLI",
               "support_source_changed": False,
               "test_labels_used": False, "test_errors_read": False,
               "selection_by_support_or_error": False,
               "estimator_fits": 0, "estimator_loads": 0, "estimator_predictions": 0,
               "input_bindings": {
                   Path(path).resolve().as_posix(): bindings[Path(path).resolve().as_posix()]
                   for path in [input_root / name for name in
                                ("records_revised.csv", "pairs_revised.csv",
                                 "record_source_group_map.csv", "fingerprints.csv")]
                   + [Path(plan["structural_support_definition_reference"]),
                      Path(plan["census_definition_reference"])]}}
    (output / "METADATA_BINDING.json").write_text(
        json.dumps(binding, indent=2) + "\n", encoding="utf-8", newline="\n")
    return pair_map, census, support
