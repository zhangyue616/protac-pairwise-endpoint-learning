"""Execute the fixed-prediction component resampling contract."""
from pathlib import Path
import csv
import json
import math
import os
import sys

# Trailing digits of the weighted dot products depend on the OpenBLAS kernel; a value set in the shell wins.
os.environ.setdefault("OPENBLAS_CORETYPE", "Nehalem")

import numpy as np

ROOT = None
BASE = None
SEED, DRAWS = 20261027, 2000
RESULT = None


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def csv_out(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


def display_three_significant(value):
    if value == 0:
        return "0"
    return f"{value:.{max(0, 2 - math.floor(math.log10(abs(value))))}f}"


def pooled_slope(weight, stats):
    n, sx, sy, sxx, sxy = (float(weight @ stats[:, i]) for i in range(5))
    if n < 2:
        return None, "weighted_observations_below_two"
    denominator = sxx - sx * sx / n
    if denominator <= 0:
        return None, "zero_or_nonpositive_weighted_predictor_variance"
    slope = (sxy - sx * sy / n) / denominator
    if not math.isfinite(slope):
        return None, "nonfinite_slope"
    return slope, ""


def direct_slope(x, response, weights):
    total = math.fsum(map(float, weights))
    mx = math.fsum(float(w) * float(v) for w, v in zip(weights, x)) / total
    my = math.fsum(float(w) * float(v) for w, v in zip(weights, response)) / total
    numerator = math.fsum(float(w) * (float(a) - mx) * (float(b) - my) for w, a, b in zip(weights, x, response))
    denominator = math.fsum(float(w) * (float(a) - mx) ** 2 for w, a in zip(weights, x))
    return numerator / denominator


def main():
    import argparse
    global ROOT, BASE, RESULT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True, type=Path, help='OLS_PARAMETERS.json and inputs/ saved XGB E1 predictions')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    BASE, ROOT = args.input_dir.resolve(), args.output_dir.resolve()
    if ROOT.exists() and any(ROOT.iterdir()):
        raise SystemExit('Output directory must be empty.')
    ROOT.mkdir(parents=True, exist_ok=True)
    RESULT = ROOT / 'SLOPE_INTERVAL_RESULTS.json'
    if RESULT.exists():
        raise SystemExit("Results already exist: do not rerun the locked computation by default.")
    old = json.loads((BASE / "OLS_PARAMETERS.json").read_text(encoding="utf-8"))
    original = {(p["protocol"], p["representation"]): p for p in old["panels"]}
    results, checks, provenance = [], [], []
    for protocol, expected_n, expected_k, cluster_field in [
        ("internal", 3450, 38, "dependency_component"),
        ("loco", 5804, 43, "heldout_component_id"),
    ]:
        source = BASE / "inputs" / f"{protocol}_XGB_E1_predictions.csv"
        with source.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if {r["representation"] for r in rows} != {"binary", "count"}:
            raise ValueError("Unexpected representation values")
        if any(not r[cluster_field] for r in rows):
            raise ValueError("Missing component; stop rather than exclude rows")
        components = sorted({r[cluster_field] for r in rows})
        if len(components) != expected_k:
            raise ValueError("Component population differs from locked contract")
        lookup = {component: j for j, component in enumerate(components)}
        # Each protocol independently starts at the same locked seed.
        rng = np.random.default_rng(SEED)
        weights = np.array([
            np.bincount(rng.integers(0, expected_k, expected_k), minlength=expected_k)
            for _ in range(DRAWS)
        ], dtype=np.int64)
        csv_out(ROOT / f"{protocol}_component_multiplicities.csv", ["draw_1based"] + components,
                ([j + 1] + weight.tolist() for j, weight in enumerate(weights)))
        save_json(ROOT / f"{protocol}_component_order.json", components)
        for representation in ("binary", "count"):
            selected = [r for r in rows if r["representation"] == representation]
            if len(selected) != expected_n or {r[cluster_field] for r in selected} != set(components):
                raise ValueError("Saved population differs from locked contract")
            x = np.array([float(r["true_delta"]) for r in selected])
            response = np.array([float(r["calibrated_prediction"]) for r in selected])
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(response)):
                raise ValueError("Nonfinite source value; stop rather than exclude observations")
            indices = np.array([lookup[r[cluster_field]] for r in selected], dtype=np.int64)
            # Fixed full-sample centering preserves the free-intercept estimator.
            cx = x - math.fsum(map(float, x)) / len(x)
            cy = response - math.fsum(map(float, response)) / len(response)
            stats = np.array([
                [int(np.sum(indices == j)),
                 math.fsum(map(float, cx[indices == j])),
                 math.fsum(map(float, cy[indices == j])),
                 math.fsum(float(v) ** 2 for v in cx[indices == j]),
                 math.fsum(float(a) * float(b) for a, b in zip(cx[indices == j], cy[indices == j]))]
                for j in range(expected_k)
            ], dtype=float)
            baseline = original[(protocol, representation)]
            point = float(baseline["slope_b"])
            reproduced = direct_slope(x, response, np.ones(len(x)))
            if not math.isclose(point, reproduced, rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError("Existing OLS point estimate failed reproduction")
            checks.append({"protocol": protocol, "representation": representation,
                           "check": "original_point_estimate", "expected": point,
                           "actual": reproduced, "absolute_difference": abs(point - reproduced), "status": "PASS"})
            draws = [pooled_slope(weight, stats) for weight in weights]
            csv_out(ROOT / f"{protocol}_{representation}_slope_draws.csv",
                    ["draw_1based", "slope", "status", "invalid_reason"],
                    ([j + 1, slope if slope is not None else "NA", "valid" if slope is not None else "invalid", reason]
                     for j, (slope, reason) in enumerate(draws)))
            finite = np.array([slope for slope, _ in draws if slope is not None])
            if not len(finite):
                raise ValueError("No valid draws; interval cannot be produced")
            low, high = map(float, np.percentile(finite, [2.5, 97.5], method="linear"))
            for draw_index in (0, 999, 1999):
                estimate, reason = draws[draw_index]
                if estimate is None:
                    checks.append({"protocol": protocol, "representation": representation,
                                   "check": "direct_weighted_spotcheck", "draw_1based": draw_index + 1,
                                   "status": "NOT_IDENTIFIABLE", "reason": reason})
                    continue
                direct = direct_slope(x, response, weights[draw_index][indices])
                if not math.isclose(estimate, direct, rel_tol=1e-11, abs_tol=1e-13):
                    raise ValueError("Component-summary OLS disagrees with direct observation-weighted OLS")
                checks.append({"protocol": protocol, "representation": representation,
                               "check": "direct_weighted_spotcheck", "draw_1based": draw_index + 1,
                               "component_summary_slope": estimate, "direct_observation_slope": direct,
                               "absolute_difference": abs(estimate - direct), "status": "PASS"})
            invalid_reasons = {}
            for slope, reason in draws:
                if slope is None:
                    invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
            results.append({"protocol": protocol, "representation": representation,
                            "observations": len(x), "components": expected_k,
                            "point_estimate": reproduced, "interval_low": low, "interval_high": high,
                            "display_point_3sig": display_three_significant(reproduced),
                            "display_interval_3sig": [display_three_significant(low), display_three_significant(high)],
                            "requested_draws": DRAWS, "valid_draws": len(finite),
                            "invalid_draws": DRAWS - len(finite), "invalid_reasons": invalid_reasons})
        provenance.append({"protocol": protocol, "saved_predictions": str(source),
                           "component_field": cluster_field, "components": components,
                           "seed_independently_initialized": SEED,
                           "same_draw_weights_for_binary_and_count": True,
                           "internal_all_repeats_inherit_component_multiplicity": protocol == "internal"})
    payload = {
        "date": "2026-10-27", "status": "LOCKED_COMPUTATION_COMPLETE_TARGETED_CHECKS_PASS",
        "script": str(Path(__file__).resolve()),
        "python": sys.version, "numpy": np.__version__, "seed": SEED,
        "draws_per_protocol": DRAWS, "interval_percentiles": [2.5, 97.5], "percentile_method": "linear",
        "table_header": "95% conditional component-resampling interval (fixed predictions)",
        "estimator": "Pooled observation-weighted free-intercept OLS: calibrated_prediction = a + b * true_delta",
        "mapping_evidence": "Saved prediction tables must already contain the frozen source-component assignments.",
        "limitations": "Predictions, fitted models, selection and calibration are fixed. These conditional intervals omit training, selection, data-construction and unresolved source-family uncertainty and do not establish nominal population coverage or significance of slope differences.",
        "inputs": provenance, "rows": results, "targeted_acceptance_checks": checks,
        "not_performed": ["model training", "prediction", "recalibration", "old MAE or CI recomputation", "slope-difference test", "mapping recensus"],
    }
    save_json(RESULT, payload)
    csv_out(ROOT / "FOUR_SLOPE_TABLE.csv",
            ["protocol", "representation", "slope", "interval_low", "interval_high", "valid_draws", "requested_draws"],
            ([r["protocol"], r["representation"], r["point_estimate"], r["interval_low"], r["interval_high"], r["valid_draws"], r["requested_draws"]] for r in results))
    print(json.dumps({"results": results, "targeted_checks": len(checks), "status": payload["status"]}, indent=2))


if __name__ == "__main__":
    main()
