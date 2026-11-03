"""Compare reader outputs with released tables using only the standard library.

Run from the repository root, after running the four replay entry points::

    python code/aggregation/compare_replay.py . ../main-tables \
        ../primary-intervals ../component-tables ../slope-intervals

REPLAY_COMPARISON.csv supplies the relative table paths. Only its ``not_run``
rows control skipping; recorded results and differences never decide a pass.
Numeric values use an absolute tolerance (default 1e-12). CSV headers, row order,
text and missingness must match. JSON objects ignore key ordering; arrays retain
their order. Byte differences alone do not fail a parsed comparison.
"""

import argparse
import csv
import json
import math
from pathlib import Path, PurePosixPath


MANIFEST_COLUMNS = [
    "table", "entry_point", "openblas_kernel", "result",
    "max_abs_difference", "notes",
]
PASS_RESULTS = {"byte_identical", "within_tolerance_only"}
EXPECTED_COMPARED = 28  # rows of REPLAY_COMPARISON.csv not marked not_run


def _detail():
    return {
        "structure": "same",
        "numeric_cells_differ": 0,
        "text_cells_differ": 0,
        "missingness_mismatches": 0,
        "nonfinite_mismatches": 0,
        "max_abs_difference": 0.0,
    }


def _difference(detail, kind, location):
    detail[kind] += 1
    detail.setdefault("first_difference", location)


def _number_difference(a, b, detail, location):
    # NaNs represent missing CSV numeric values. Their positions must agree.
    nan_a = isinstance(a, float) and math.isnan(a)
    nan_b = isinstance(b, float) and math.isnan(b)
    if nan_a or nan_b:
        if nan_a != nan_b:
            _difference(detail, "missingness_mismatches", location)
        return
    # Never let infinities, overflow, or NaN arithmetic silently pass a bound.
    if any(isinstance(x, float) and not math.isfinite(x) for x in (a, b)):
        _difference(detail, "nonfinite_mismatches", location)
        return
    if a == b:
        return
    try:
        gap = abs(a - b)
    except OverflowError:
        _difference(detail, "nonfinite_mismatches", location)
        return
    if isinstance(gap, float) and not math.isfinite(gap):
        _difference(detail, "nonfinite_mismatches", location)
        return
    _difference(detail, "numeric_cells_differ", location)
    detail["max_abs_difference"] = max(detail["max_abs_difference"], gap)


def _as_float(value):
    try:
        return float(value)
    except ValueError:
        return None


def compare_csv(rebuilt, released):
    """Compare aligned CSV cells, checking the entire shape before arithmetic."""
    with rebuilt.open(encoding="utf-8-sig", newline="") as handle:
        rows_a = list(csv.reader(handle, strict=True))
    with released.open(encoding="utf-8-sig", newline="") as handle:
        rows_b = list(csv.reader(handle, strict=True))
    detail = _detail()
    problem = None
    if len(rows_a) != len(rows_b):
        problem = "row count differs"
    elif rows_a and rows_a[0] != rows_b[0]:
        problem = "headers or column order differ"
    elif rows_a:
        width = len(rows_a[0])
        for index, (row_a, row_b) in enumerate(zip(rows_a, rows_b), 1):
            if len(row_a) != width or len(row_b) != width:
                problem = f"row {index}: width differs from the header"
                break
    if problem:
        detail.update(structure="differs", max_abs_difference=None,
                      first_difference=problem)
        return detail
    for index, (row_a, row_b) in enumerate(zip(rows_a[1:], rows_b[1:]), 2):
        for column, (cell_a, cell_b) in enumerate(zip(row_a, row_b), 1):
            if cell_a == cell_b:
                continue
            location = f"row {index}, column {column}"
            a, b = _as_float(cell_a), _as_float(cell_b)
            if a is None or b is None:
                _difference(detail, "text_cells_differ", location)
            else:
                _number_difference(a, b, detail, location)
    return detail


def _reject_constant(value):
    raise ValueError(f"nonstandard JSON constant: {value}")


def _finite_json_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON number exceeds the finite float range")
    return number


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def compare_json(rebuilt, released):
    """Compare standard JSON recursively, preserving types and array ordering."""
    def read(path):
        with path.open(encoding="utf-8-sig") as handle:
            return json.load(handle, parse_constant=_reject_constant,
                             parse_float=_finite_json_float,
                             object_pairs_hook=_unique_object)

    detail = _detail()

    def structural(location):
        detail["structure"] = "differs"
        detail.setdefault("first_difference", location)

    def visit(a, b, location):
        numeric_a = type(a) in (int, float)
        numeric_b = type(b) in (int, float)
        if numeric_a and numeric_b:
            _number_difference(a, b, detail, location)
        elif type(a) is not type(b):
            structural(location + ": value types differ")
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                structural(location + ": object keys differ")
                return
            for key in a:
                visit(a[key], b[key], f"{location}[{key!r}]")
        elif isinstance(a, list):
            if len(a) != len(b):
                structural(location + ": array lengths differ")
                return
            for index, (value_a, value_b) in enumerate(zip(a, b)):
                visit(value_a, value_b, f"{location}[{index}]")
        elif a != b:
            _difference(detail, "text_cells_differ", location)

    visit(read(rebuilt), read(released), "$")
    return detail


def _classify(detail, byte_identical, tolerance):
    if detail["structure"] != "same":
        return "structural_mismatch"
    if detail["text_cells_differ"]:
        return "text_mismatch"
    if detail["missingness_mismatches"]:
        return "missingness_mismatch"
    if detail["nonfinite_mismatches"]:
        return "nonfinite_mismatch"
    if detail["max_abs_difference"] > tolerance:
        return "numerical_mismatch"
    return "byte_identical" if byte_identical else "within_tolerance_only"


def read_manifest(repo):
    with (repo / "code/aggregation/REPLAY_COMPARISON.csv").open(
            encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames != MANIFEST_COLUMNS:
            raise ValueError("comparison manifest has unexpected columns")
        rows = list(reader)
    seen = set()
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("comparison manifest has a malformed row")
        table = row["table"]
        path = PurePosixPath(table)
        if not table or path.is_absolute() or ".." in path.parts or "\\" in table or ":" in table:
            raise ValueError("manifest table paths must be relative POSIX paths")
        if table in seen:
            raise ValueError(f"duplicate manifest table: {table}")
        seen.add(table)
    return rows


def compare_outputs(repo, outputs, rows, tolerance):
    report = []
    for row in rows:
        table = row["table"]
        released = repo / table
        item = {"table": table, "filename": released.name,
                "entry_point": row["entry_point"], "max_abs_difference": None}
        status = None
        if row["result"] == "not_run":
            status = "not_run"
        elif not released.is_file():
            status = "missing_reference"
        else:
            found = {directory / released.name for directory in outputs
                     if (directory / released.name).is_file()}
            if not found:
                status = "missing_output"
            elif len(found) != 1:
                status = "ambiguous_output"
            else:
                rebuilt = found.pop()
                try:
                    if released.suffix.lower() == ".csv":
                        detail = compare_csv(rebuilt, released)
                    elif released.suffix.lower() == ".json":
                        detail = compare_json(rebuilt, released)
                    else:
                        raise ValueError("only CSV and JSON tables are supported")
                    identical = rebuilt.read_bytes() == released.read_bytes()
                    status = _classify(detail, identical, tolerance)
                    item.update(detail)
                except (OSError, ValueError, csv.Error, RecursionError) as error:
                    status = "invalid_format"
                    item["error"] = str(error)
        item.update(status=status, result=status)
        report.append(item)

    counts = {}
    for item in report:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    passed = sum(counts.get(status, 0) for status in PASS_RESULTS)
    skipped = counts.get("not_run", 0)
    summary = {
        "total": len(report), "compared": len(report) - skipped,
        "passed": passed, "failed": len(report) - skipped - passed,
        "skipped": skipped, "tolerance": tolerance,
        "max_abs_difference": max(
            (item["max_abs_difference"] for item in report
             if item["max_abs_difference"] is not None), default=0.0),
        "counts": counts,
    }
    return {"summary": summary, "rows": report}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("outputs", nargs="+", type=Path)
    parser.add_argument("--tolerance", "--tol", type=float, default=1e-12,
                        help="absolute numeric tolerance (default: 1e-12)")
    parser.add_argument("--json", type=Path, metavar="FILE",
                        help="write actual results and summary as JSON")
    args = parser.parse_args(argv)
    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("tolerance must be finite and nonnegative")
    checkout = args.repo.resolve()
    outputs = {path.resolve() for path in args.outputs}
    if any(path == checkout or checkout in path.parents for path in outputs):
        parser.error("output directories must be outside the checkout")
    try:
        rows = read_manifest(args.repo)
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))
    report = compare_outputs(args.repo, outputs, rows, args.tolerance)
    for item in report["rows"]:
        maximum = item["max_abs_difference"]
        explanation = item.get("error", item.get("first_difference", ""))
        print(f"{item['status']:<24} max_abs={str(maximum):<24} {item['table']}"
              + (f" ({explanation})" if explanation else ""))
    print(json.dumps(report["summary"], sort_keys=True, allow_nan=False))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                             encoding="utf-8")
    compared = report["summary"]["compared"]
    if compared != EXPECTED_COMPARED:
        print(f"expected {EXPECTED_COMPARED} compared files, found {compared}")
        return 1
    return 1 if report["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
