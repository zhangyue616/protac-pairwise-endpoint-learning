"""Replay four component reader tables using unchanged archived AST statements.

Provide the r6_before_companion_filter directory written by replay_main_tables.
The archived CSV reader, scenario filter and writer are executed unchanged.
Only input staging, AST selection and manifest-defined publication are new.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import shutil
from statistics import fmean
import sys

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ARCHIVE_SHA256 = "567385C0962CBC9ACA496960EDD79230A2AC7B4E7A7185D8F3FDCD24FE3A16CE"
TABLES = {
    "component_loco_metrics.csv": {
        "ranges": [(126, 141)], "archive_output": "R6_Table3.csv",
        "inputs": ["MACRO_METRICS.csv", "ZERO_BY_GROUP.csv", "LEARNER_CONTRAST_SUMMARY.csv"],
    },
    "component_loco_contrasts.csv": {
        "ranges": [(155, 156), (159, 159), (170, 178)], "archive_output": "R6_S27_contrasts.csv",
        "inputs": ["GROUP_AVERAGE_CONTRASTS.csv"],
    },
    "component_loco_summary.csv": {
        "ranges": [(129, 130), (179, 196)], "archive_output": "R6_S28_summary.csv",
        "inputs": ["LEARNER_CONTRAST_SUMMARY.csv", "PRIMARY_SUMMARY.csv",
                   "SCALE_DIAGNOSTIC_ALL_43.csv", "GROUP_LEARNER_CONTRASTS.csv"],
    },
    "component_loco_sensitivity.csv": {
        "ranges": [(179, 179), (197, 211)], "archive_output": "R6_S29_sensitivity.csv",
        "inputs": ["PRIMARY_SUMMARY.csv", "SENSITIVITY.csv"],
    },
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def node_identity(node):
    serialized = ast.dump(node, annotate_fields=True, include_attributes=False)
    return {"type": type(node).__name__, "first_line": node.lineno,
            "last_line": node.end_lineno, "ast_dump": serialized,
            "ast_dump_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest().upper()}


def select_statements(main_body, ranges):
    selected = []
    for first, last in ranges:
        nodes = [node for node in main_body if first <= node.lineno <= last]
        require(nodes and nodes[0].lineno == first and nodes[-1].end_lineno == last,
                "Selection must contain complete archived statements: " + str((first, last)))
        require(all(node.end_lineno <= last for node in nodes), "Selection cuts an archived statement")
        selected.extend(nodes)
    return selected


def execute_unchanged(nodes, namespace, filename):
    identities = [node_identity(node) for node in nodes]
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(filename), "exec"), namespace)
    require(identities == [node_identity(node) for node in nodes], "Archived AST changed during execution")
    return identities


def publish(source, name, manifest, namespace, output):
    entries = [item for item in manifest["files"]
               if item["path"] == "results/current_companion/" + name]
    require(len(entries) == 1, "Missing/ambiguous publication manifest entry: " + name)
    entry = entries[0]
    removed = entry["removed_columns"]
    if entry["transformation"] == "exact_byte_copy":
        require(not removed, "Exact-copy publication cannot remove columns")
        shutil.copyfile(source, output / name)
    else:
        require(entry["transformation"] == "drop_non_scientific_path_columns"
                and name == "component_loco_metrics.csv" and removed == ["source"],
                "Unsupported publication transformation: " + name)
        rows = namespace["readc"](source)
        require(all(set(removed) <= set(row) for row in rows), "Publication column missing")
        rows = [{key: value for key, value in row.items() if key not in removed} for row in rows]
        original_output = namespace["OUT"]
        namespace["OUT"] = output
        try:
            namespace["writec"](name, rows)
        finally:
            namespace["OUT"] = original_output
    with (output / name).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == entry["columns"], "Publication column order drift: " + name)
        require(len(list(reader)) == entry["rows"], "Publication row population drift: " + name)
    return {"transformation": entry["transformation"], "removed_columns": removed,
            "sha256": digest(output / name)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r6-results", required=True, type=Path,
                        help="Original R6 CSV serialization before the companion pandas read/filter/write")
    parser.add_argument("--group-support", type=Path,
                        default=REPO / "results/current_companion/S26_group_support.csv",
                        help="Public component support metadata; only component keys are used")
    parser.add_argument("--output", required=True, type=Path, help="New directory outside this repository")
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), "Output directory must not already exist")
    require(not output.is_relative_to(REPO), "Output must be outside the repository")
    archive = HERE / "extract_mapping.py"
    require(digest(archive) == ARCHIVE_SHA256, "Archived extraction source SHA mismatch")
    tree = ast.parse(archive.read_text(encoding="utf-8"), filename=str(archive))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    helpers = [functions[name] for name in ("readc", "writec", "r6")]
    body = functions["main"].body
    selections = {name: select_statements(body, spec["ranges"]) for name, spec in TABLES.items()}
    manifest_path = REPO / "results/result_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    upstream = args.r6_results.resolve(strict=True)
    required = sorted({name for spec in TABLES.values() for name in spec["inputs"]})
    source_paths = [upstream / name for name in required]
    source_paths.append(args.group_support.resolve(strict=True))
    require(all(path.is_file() for path in source_paths), "Required R6 upstream CSV or support metadata missing")
    require(not any(path.is_relative_to(REPO / "results/current_companion") for path in source_paths[:-1]),
            "R6 inputs must be replay upstream CSVs, not retained target tables")
    output.mkdir(parents=True)
    staged = output / "_r6_input"
    (staged / "results").mkdir(parents=True)
    (staged / "corrected_metadata").mkdir()
    for path in source_paths[:-1]:
        shutil.copyfile(path, staged / "results" / path.name)
    shutil.copyfile(source_paths[-1], staged / "corrected_metadata/group_support.csv")
    receipt = {
        "status": "COMPONENT_READER_AST_REPLAY_COMPLETE", "source_sha256": ARCHIVE_SHA256,
        "source": str(archive), "wrapper_sha256": digest(__file__),
        "helper_ast_identity": [node_identity(node) for node in helpers],
        "input_bindings": {str(path): digest(path) for path in source_paths},
        "publication_manifest_sha256": digest(manifest_path),
        "input_boundary": "R6 serialization before companion pandas default read/filter/write",
        "parsing": "Unchanged csv.DictReader(utf-8-sig, newline='') helper; no pandas parsing",
        "statistical_changes": [], "fits": 0, "new_predictions": 0, "bootstrap_calls": 0,
        "tie_threshold": "Unchanged archived 1e-12 comparisons in summary; no replay tolerance",
        "tables": {},
    }
    for name, spec in TABLES.items():
        original_dir = output / "_archived_outputs" / name.removesuffix(".csv")
        original_dir.mkdir(parents=True)
        namespace = {"__builtins__": __builtins__, "csv": csv, "fmean": fmean,
                     "OUT": original_dir, "R6": staged}
        execute_unchanged(helpers, namespace, archive)
        identities = execute_unchanged(selections[name], namespace, archive)
        publication = publish(original_dir / spec["archive_output"], name, manifest, namespace, output)
        receipt["tables"][name] = {"status": "REPLAYED", "source_ranges": spec["ranges"],
                                  "ast_identity": identities, "publication": publication}
    receipt["tables"]["component_loco_support.csv"] = {
        "status": "SKIPPED_UNPUBLISHED_DEPENDENCY", "source_ranges": [[158, 169]],
        "reason": "Archived support extraction requires record_source_group_map.csv.normalized_source_id; "
                  "that identifier is not supplied by the public mapping. c2_source_id is not substituted.",
    }
    (output / "COMPONENT_REPLAY_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": receipt["status"], "replayed": len(TABLES), "skipped": 1,
                      "output": str(output), "fits": 0, "new_predictions": 0, "bootstrap_calls": 0}))


if __name__ == "__main__":
    main()
