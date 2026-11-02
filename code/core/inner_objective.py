"""Single replacement point for coupled training-only OOF scale and loss.

The controller-authorized draft assigns equal total weight to each component
with OOF pairs. Scale and candidate loss use the same population and weights.
Pair-pooled MAE describes these same predictions and never selects a candidate.
This pure module does not fit models or provide execution authority.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

import numpy as np

SUPPORTED_OBJECTIVES = frozenset({"component_macro"})
EXPECTED_POSITIVE_OOF_COMPONENTS = 42
CANDIDATE_TIE_TOLERANCE = 1e-12


def validate_objective(name: str) -> str:
    """Reject a missing or unsupported objective before any scientific fit."""
    if not isinstance(name, str) or name not in SUPPORTED_OBJECTIVES:
        raise RuntimeError(f"unsupported draft inner objective: {name!r}")
    return name


def _population(y: Any, raw: Any, component_ids: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    truth = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(raw, dtype=np.float64)
    components = np.asarray(component_ids)
    if truth.ndim != 1 or prediction.shape != truth.shape or components.shape != truth.shape or not len(truth):
        raise RuntimeError("invalid inner OOF objective population")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("inner OOF truth and prediction must be finite")
    if any(value is None or not str(value).strip() or str(value).lower() == "nan" for value in components):
        raise ValueError("inner OOF component identity must be known")
    _, codes = np.unique(components.astype(str), return_inverse=True)
    counts = np.bincount(codes)
    return truth, prediction, codes, counts


def validate_oof_population(pair_ids: Any, component_ids: Any, expected_pair_ids: Any,
                            expected_pair_component_ids: Any, *,
                            expected_components: int = EXPECTED_POSITIVE_OOF_COMPONENTS) -> np.ndarray:
    """Bind each canonical pair once to the complete frozen expected scope."""
    pairs, components = np.asarray(pair_ids), np.asarray(component_ids)
    expected_pairs, expected_labels = np.asarray(expected_pair_ids), np.asarray(expected_pair_component_ids)
    if pairs.ndim != 1 or not len(pairs) or any(arr.shape != pairs.shape for arr in
            (components, expected_pairs, expected_labels)):
        raise RuntimeError("empty or mismatched OOF population")
    if any(value is None or not str(value).strip() or str(value).lower() == "nan"
           for array in (pairs, components, expected_pairs, expected_labels) for value in array):
        raise RuntimeError("missing OOF pair or component identity")
    if len(np.unique(pairs.astype(str))) != len(pairs) or len(np.unique(expected_pairs.astype(str))) != len(expected_pairs):
        raise RuntimeError("canonical OOF pair must occur exactly once")
    if not np.array_equal(pairs, expected_pairs) or not np.array_equal(components, expected_labels):
        raise RuntimeError("OOF pair/component mapping differs from frozen expected population")
    _, codes, counts = np.unique(expected_labels.astype(str), return_inverse=True, return_counts=True)
    if len(counts) != expected_components or np.any(counts <= 0):
        raise RuntimeError(f"OOF requires exactly {expected_components} nonempty expected components")
    weights = 1.0 / (len(counts) * counts[codes].astype(np.float64))
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise RuntimeError("invalid OOF loss weights")
    return weights


def _weights(codes: np.ndarray, counts: np.ndarray, loss_weights: Any | None,
             expected_components: int | None) -> np.ndarray:
    if expected_components is not None and len(counts) != expected_components:
        raise RuntimeError(f"OOF requires exactly {expected_components} components")
    canonical = 1.0 / (len(counts) * counts[codes].astype(np.float64))
    weights = canonical if loss_weights is None else np.asarray(loss_weights, dtype=np.float64)
    if weights.shape != codes.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("OOF weights must be finite, positive and aligned")
    if not np.array_equal(weights, canonical):
        raise ValueError("OOF weights must be frozen equal-component 1/(K*n_g) weights")
    return weights


def nonnegative_component_l1_scale(y: Any, raw: Any, component_ids: Any, *,
                                  loss_weights: Any | None = None,
                                  expected_components: int | None = None) -> float:
    """Select the exact lower L1 minimizer, then return its binary64 rounding.

    Component counts include every OOF pair, including p=0 constant terms.
    For nonzero p, ratios y/p have weights abs(p)/n_component. The omitted
    common 1/G factor cannot change the exact minimizer. All-zero p returns c=0.
    Finite binary64 inputs are interpreted as exact rationals for selection.
    The returned float need not minimize the exact loss over real or binary64
    scales. Reported candidate losses retain floating-point arithmetic.
    """
    truth, prediction, codes, counts = _population(y, raw, component_ids)
    _weights(codes, counts, loss_weights, expected_components)
    active = prediction != 0.0
    if not np.any(active):
        return 0.0
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        ratios = truth[active] / prediction[active]
        # Omit the common 1/K; scale by max|q| before summation/division.
        magnitude = np.abs(prediction[active])
        weights = (magnitude / float(np.max(magnitude))) / counts[codes[active]]
    if not np.isfinite(ratios).all():
        raise ValueError("inner OOF y / p produced a non-finite ratio")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("active ratio weights became non-finite or underflowed to zero")
    # Keep the float ratio/weight guards above, but decide the minimizer using
    # the exact rational values represented by the finite binary64 inputs.
    # Binary64 denominators are powers of two. A common slope denominator and
    # component-count LCM make every abs(p)/n_component weight an exact integer
    # multiple of one common positive factor; no rescaling or rounded prefix
    # can turn an inclusive exact-half boundary into a strict crossing.
    slopes = [float(value).as_integer_ratio() for value in prediction[active]]
    slope_denominator = max(denominator for _, denominator in slopes)
    count_multiple = math.lcm(*(int(count) for count in counts))
    exact_weights = [abs(numerator) * (slope_denominator // denominator)
                     * (count_multiple // int(counts[code]))
                     for (numerator, denominator), code in zip(slopes, codes[active])]
    exact_ratios = [Fraction(float(value)) / Fraction(numerator, denominator)
                    for value, (numerator, denominator) in zip(truth[active], slopes)]
    # Sorting exact ratios also distinguishes unequal ratios that round to the
    # same float. Duplicate exact ratios need no merge: their boundary is the
    # same ratio regardless of which duplicate first reaches inclusive half.
    # float() below rounds the selected exact minimizer for the binary64 API;
    # this conversion does not generally preserve exact optimality.
    order = sorted(range(len(exact_ratios)), key=exact_ratios.__getitem__)
    total_weight = sum(exact_weights)
    if total_weight <= 0:
        raise ValueError("active ratio total weight is invalid")
    cumulative = 0
    weighted_median = float(exact_ratios[order[-1]])
    for index in order:
        cumulative += exact_weights[index]
        if 2 * cumulative >= total_weight:
            weighted_median = float(exact_ratios[index])
            break
    c = max(0.0, weighted_median)
    if not math.isfinite(c):
        raise ValueError("component L1 scale is non-finite")
    return c


def evaluate_inner_oof(y: Any, raw: Any, component_ids: Any, objective: str, *,
                       loss_weights: Any | None = None,
                       expected_components: int | None = None) -> dict[str, Any]:
    validate_objective(objective)
    truth, prediction, codes, counts = _population(y, raw, component_ids)
    weights = _weights(codes, counts, loss_weights, expected_components)
    c = nonnegative_component_l1_scale(truth, prediction, component_ids,
                                      loss_weights=weights, expected_components=expected_components)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        scaled = c * prediction
        raw_error, calibrated_error = np.abs(truth - prediction), np.abs(truth - scaled)
    if not all(np.isfinite(array).all() for array in (scaled, raw_error, calibrated_error)):
        raise ValueError("scaled prediction or OOF loss is non-finite")
    macro_raw = math.fsum(float(value) for value in weights * raw_error)
    macro_calibrated = math.fsum(float(value) for value in weights * calibrated_error)
    pair_raw = math.fsum(float(value) for value in raw_error / len(truth))
    pair_calibrated = math.fsum(float(value) for value in calibrated_error / len(truth))
    if not all(math.isfinite(value) for value in (macro_raw, macro_calibrated, pair_raw, pair_calibrated)):
        raise ValueError("OOF summary loss is non-finite")
    return {"scale_c": c,
            "inner_oof_raw_component_macro_mae": macro_raw,
            "inner_oof_calibrated_component_macro_mae": macro_calibrated,
            "inner_oof_raw_pair_pooled_mae": pair_raw,
            "inner_oof_calibrated_pair_pooled_mae": pair_calibrated,
            "selection_objective": objective,
            "selection_score": macro_calibrated,
            "inner_oof_components": int(len(counts)),
            "component_weight_rule": "a_i=1/(K*n_g); each component total 1/K",
            "all_zero_raw_prediction": bool(np.all(prediction == 0.0))}


def select_candidate(summaries: Any, *, expected_candidates: int = 8) -> dict[str, Any]:
    """Complete finite candidate set, absolute 1e-12 score tie, then lex ID."""
    rows = list(summaries)
    if len(rows) != expected_candidates or expected_candidates <= 0:
        raise RuntimeError("candidate selection requires the complete expected set")
    identifiers = [row.get("candidate_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in identifiers) or len(set(identifiers)) != len(rows):
        raise RuntimeError("candidate IDs must be known and unique")
    for row in rows:
        score, c = float(row["selection_score"]), float(row["scale_c"])
        if not math.isfinite(score) or score < 0 or not math.isfinite(c) or c < 0:
            raise RuntimeError("candidate score and nonnegative scale must be finite")
    best = min(float(row["selection_score"]) for row in rows)
    return min((row for row in rows if float(row["selection_score"]) <= best + CANDIDATE_TIE_TOLERANCE),
               key=lambda row: row["candidate_id"])
