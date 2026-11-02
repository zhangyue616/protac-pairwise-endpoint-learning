"""Small, deterministic model adapters used by the Route-S runner.

The caller owns hyper-parameter selection and train/test splitting.  This
module only enforces the invariants that are shared by all Route-S fits.
"""

from __future__ import annotations

import json
import math
from time import perf_counter
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


_SUPPORTED_LEARNERS = {"ridge", "random_forest", "xgboost"}
_TREE_LEARNERS = {"random_forest", "xgboost"}


def _normalise_learner(learner: str) -> str:
    if not isinstance(learner, str):
        raise TypeError("learner must be a string")
    name = learner.strip().lower()
    if name not in _SUPPORTED_LEARNERS:
        allowed = ", ".join(sorted(_SUPPORTED_LEARNERS))
        raise ValueError(f"unsupported learner {learner!r}; expected one of: {allowed}")
    return name


def _as_finite_matrix(X: Any, *, name: str = "X") -> np.ndarray:
    try:
        array = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric two-dimensional array") from exc
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have at least one row and one feature")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _as_finite_vector(values: Any, *, name: str, rows: int | None = None) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric one-dimensional array") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if rows is not None and array.shape[0] != rows:
        raise ValueError(f"{name} has {array.shape[0]} rows; expected {rows}")
    if rows is None and array.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _python_block_key(value: Any) -> Hashable:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        raise ValueError("block_ids must not contain missing values")
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError("block_ids must not contain non-finite values")
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError("each block id must be hashable") from exc
    return value


def _factorise_blocks(
    block_ids: Sequence[Any] | np.ndarray,
    *,
    rows: int,
    require_at_least_two: bool,
) -> tuple[np.ndarray, list[Hashable], np.ndarray]:
    ids = np.asarray(block_ids, dtype=object)
    if ids.ndim != 1:
        raise ValueError("block_ids must be one-dimensional")
    if ids.shape[0] != rows:
        raise ValueError(f"block_ids has {ids.shape[0]} rows; expected {rows}")

    index_by_key: dict[Hashable, int] = {}
    keys: list[Hashable] = []
    codes = np.empty(rows, dtype=np.int64)
    for row, raw_key in enumerate(ids):
        key = _python_block_key(raw_key)
        code = index_by_key.get(key)
        if code is None:
            code = len(keys)
            index_by_key[key] = code
            keys.append(key)
        codes[row] = code

    counts = np.bincount(codes, minlength=len(keys)).astype(np.int64, copy=False)
    if require_at_least_two and np.any(counts < 2):
        singleton_keys = [keys[i] for i in np.flatnonzero(counts < 2)]
        raise ValueError(
            "all training blocks must contain at least two records; "
            f"singleton blocks found: {singleton_keys!r}"
        )
    return codes, keys, counts


def _group_means(values: np.ndarray, codes: np.ndarray, counts: np.ndarray) -> np.ndarray:
    sums = np.bincount(codes, weights=values, minlength=counts.shape[0])
    return sums / counts


def _find_first_device(config: Any) -> str | None:
    if isinstance(config, Mapping):
        if "device" in config and isinstance(config["device"], (str, int, float)):
            return str(config["device"])
        for value in config.values():
            found = _find_first_device(value)
            if found is not None:
                return found
    elif isinstance(config, list):
        for value in config:
            found = _find_first_device(value)
            if found is not None:
                return found
    return None


def _xgboost_config_device(model: Any) -> tuple[str | None, str | None]:
    """Return ``(device, parse_error)`` without making config parsing fatal."""

    try:
        config_text = model.get_booster().save_config()
        config = json.loads(config_text)
        return _find_first_device(config), None
    except Exception as exc:  # XGBoost config formats vary across supported versions.
        return None, f"{type(exc).__name__}: {exc}"


def fit_model(
    learner: str,
    params: Mapping[str, Any] | None,
    X: Any,
    y: Any,
    seed: int,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    """Fit one supported regressor and return it with a compact fit receipt."""

    name = _normalise_learner(learner)
    X_array = _as_finite_matrix(X)
    y_array = _as_finite_vector(y, name="y", rows=X_array.shape[0])
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise TypeError("seed must be an integer")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a non-empty string")
    if params is None:
        model_params: dict[str, Any] = {}
    elif isinstance(params, Mapping):
        model_params = dict(params)
    else:
        raise TypeError("params must be a mapping or None")

    seed_int = int(seed)
    requested_device = device.strip()
    if name == "ridge":
        model_params.setdefault("fit_intercept", False)
        model_params["random_state"] = seed_int
        model = Ridge(**model_params)
    elif name == "random_forest":
        model_params["random_state"] = seed_int
        model = RandomForestRegressor(**model_params)
    else:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("xgboost is required for learner='xgboost'") from exc
        model_params["random_state"] = seed_int
        model_params["objective"] = "reg:squarederror"
        model_params["tree_method"] = "hist"
        model_params["device"] = requested_device
        model = XGBRegressor(**model_params)

    started = perf_counter()
    model.fit(X_array, y_array)
    elapsed = perf_counter() - started

    receipt: dict[str, Any] = {
        "elapsed": float(elapsed),
        "learner": name,
        "device": requested_device,
        "rows": int(X_array.shape[0]),
        "features": int(X_array.shape[1]),
    }
    if name == "xgboost":
        actual_device, parse_error = _xgboost_config_device(model)
        receipt["actual_device"] = actual_device
        receipt["xgb_booster_device"] = actual_device
        if parse_error is not None:
            receipt["xgb_device_parse_error"] = parse_error
    return model, receipt


def predict_model(model: Any, X: Any) -> np.ndarray:
    """Return finite, one-dimensional ``float64`` predictions."""

    if not hasattr(model, "predict"):
        raise TypeError("model must provide a predict method")
    X_array = _as_finite_matrix(X)
    predictions = np.asarray(model.predict(X_array), dtype=np.float64)
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        predictions = predictions[:, 0]
    if predictions.ndim != 1:
        raise ValueError("model.predict must return one value per row")
    if predictions.shape[0] != X_array.shape[0]:
        raise ValueError(
            f"model.predict returned {predictions.shape[0]} rows; "
            f"expected {X_array.shape[0]}"
        )
    if not np.isfinite(predictions).all():
        raise ValueError("model.predict returned non-finite values")
    return predictions.astype(np.float64, copy=False)


def fit_tree_e1_backfit(
    learner: str,
    params: Mapping[str, Any] | None,
    X_records: Any,
    z: Any,
    block_ids: Sequence[Any] | np.ndarray,
    seed: int,
    device: str,
    iterations: int = 5,
    before_fit: Any | None = None,
) -> dict[str, Any]:
    """Fit the E1 block-intercept/tree backfit with exactly five tree fits."""

    name = _normalise_learner(learner)
    if name not in _TREE_LEARNERS:
        raise ValueError("fit_tree_e1_backfit accepts only random_forest or xgboost")
    if not isinstance(iterations, (int, np.integer)) or isinstance(
        iterations, (bool, np.bool_)
    ):
        raise TypeError("iterations must be an integer")
    if int(iterations) != 5:
        raise ValueError("Route-S E1 backfitting requires exactly five iterations")
    if before_fit is not None and not callable(before_fit):
        raise TypeError("before_fit must be callable or None")

    X_array = _as_finite_matrix(X_records, name="X_records")
    z_array = _as_finite_vector(z, name="z", rows=X_array.shape[0])
    codes, block_keys, counts = _factorise_blocks(
        block_ids,
        rows=X_array.shape[0],
        require_at_least_two=True,
    )

    h_previous = np.zeros(X_array.shape[0], dtype=np.float64)
    iteration_receipts: list[dict[str, Any]] = []
    model: Any | None = None
    for iteration in range(1, 6):
        raw_alpha = _group_means(z_array - h_previous, codes, counts)
        centre = float(np.mean(raw_alpha[codes], dtype=np.float64))
        alpha = raw_alpha - centre
        target = z_array - alpha[codes]

        if before_fit is not None:
            before_fit(iteration, name, params)
        model, fit_receipt = fit_model(
            name,
            params,
            X_array,
            target,
            seed,
            device,
        )
        h_previous = predict_model(model, X_array)
        iteration_receipt = dict(fit_receipt)
        iteration_receipt.update(
            {
                "iteration": iteration,
                "alpha_center": centre,
                "blocks": len(block_keys),
            }
        )
        iteration_receipts.append(iteration_receipt)

    # Re-estimate the block effects against h_5.  Moving the weighted centre
    # into h_shift keeps alpha_b + h_5 + h_shift equal to raw_alpha_b + h_5.
    final_raw_alpha = _group_means(z_array - h_previous, codes, counts)
    final_centre = float(np.mean(final_raw_alpha[codes], dtype=np.float64))
    final_alpha = final_raw_alpha - final_centre
    alpha_map = {
        block_keys[index]: float(final_alpha[index]) for index in range(len(block_keys))
    }

    if model is None:  # Defensive only; the fixed five-iteration loop always assigns it.
        raise RuntimeError("E1 backfitting did not produce a fitted model")
    return {
        "model": model,
        "h_shift": final_centre,
        "alpha": alpha_map,
        "iteration_receipts": iteration_receipts,
        "total_fit_count": 5,
    }


def fit_ridge_e1_within(
    alpha: float,
    X_records: Any,
    z: Any,
    block_ids: Sequence[Any] | np.ndarray,
) -> tuple[Ridge, dict[str, Any]]:
    """Fit ridge to block-centred records; block effects are not penalised."""

    if not isinstance(alpha, (int, float, np.integer, np.floating)) or isinstance(
        alpha, (bool, np.bool_)
    ):
        raise TypeError("alpha must be a finite non-negative scalar")
    alpha_float = float(alpha)
    if not math.isfinite(alpha_float) or alpha_float < 0.0:
        raise ValueError("alpha must be a finite non-negative scalar")

    X_array = _as_finite_matrix(X_records, name="X_records")
    z_array = _as_finite_vector(z, name="z", rows=X_array.shape[0])
    codes, block_keys, counts = _factorise_blocks(
        block_ids,
        rows=X_array.shape[0],
        require_at_least_two=True,
    )

    block_count = counts.shape[0]
    X_sums = np.zeros((block_count, X_array.shape[1]), dtype=np.float64)
    np.add.at(X_sums, codes, X_array)
    X_means = X_sums / counts[:, None]
    z_means = _group_means(z_array, codes, counts)
    X_within = X_array - X_means[codes]
    z_within = z_array - z_means[codes]

    model = Ridge(alpha=alpha_float, fit_intercept=False)
    started = perf_counter()
    model.fit(X_within, z_within)
    elapsed = perf_counter() - started
    receipt: dict[str, Any] = {
        "elapsed": float(elapsed),
        "learner": "ridge",
        "device": "cpu",
        "rows": int(X_array.shape[0]),
        "features": int(X_array.shape[1]),
        "blocks": len(block_keys),
        "alpha": alpha_float,
        "fit_intercept": False,
    }
    return model, receipt


def predict_direct_antisymmetric(
    model: Any,
    Xdiff: Any,
    zero_tol: float = 1e-12,
) -> np.ndarray:
    """Predict ``0.5 * (f(x) - f(-x))`` without clipping nonzero rows.

    ``zero_tol`` is validated for API consistency, but zero-row detection is
    deliberately exact: only rows whose features are all exactly zero are
    forced to zero.
    """

    if not isinstance(zero_tol, (int, float, np.integer, np.floating)) or isinstance(
        zero_tol, (bool, np.bool_)
    ):
        raise TypeError("zero_tol must be a finite non-negative scalar")
    zero_tol_float = float(zero_tol)
    if not math.isfinite(zero_tol_float) or zero_tol_float < 0.0:
        raise ValueError("zero_tol must be a finite non-negative scalar")

    X_array = _as_finite_matrix(Xdiff, name="Xdiff")
    positive = predict_model(model, X_array)
    negative = predict_model(model, -X_array)
    prediction = 0.5 * (positive - negative)
    exact_zero_rows = np.all(X_array == 0.0, axis=1)
    prediction[exact_zero_rows] = 0.0
    if not np.isfinite(prediction).all():
        raise ValueError("antisymmetric prediction produced non-finite values")
    return prediction.astype(np.float64, copy=False)


def nonnegative_l1_scale(y: Any, p: Any) -> float:
    """Return the deterministic non-negative L1-optimal scale.

    Nonzero ``p_i`` terms give ratios ``y_i / p_i`` with weights ``|p_i|``.
    The tie rule is the lower weighted median: the first sorted ratio whose
    cumulative weight reaches at least half the total weight.  When every
    ``p_i`` is zero, the canonical minimiser is 0.
    """

    y_array = np.asarray(y, dtype=np.float64)
    p_array = np.asarray(p, dtype=np.float64)
    if y_array.ndim != 1 or p_array.ndim != 1:
        raise ValueError("y and p must be one-dimensional")
    if y_array.shape[0] != p_array.shape[0]:
        raise ValueError("y and p must have the same length")
    if not np.isfinite(y_array).all() or not np.isfinite(p_array).all():
        raise ValueError("y and p must contain only finite values")

    active = p_array != 0.0
    if not np.any(active):
        return 0.0
    ratios = y_array[active] / p_array[active]
    weights = np.abs(p_array[active])
    if not np.isfinite(ratios).all():
        raise ValueError("y / p produced a non-finite ratio")

    order = np.argsort(ratios, kind="mergesort")
    sorted_ratios = ratios[order]
    sorted_weights = weights[order]
    # Scaling avoids overflow in the total without changing weighted-median
    # comparisons. math.fsum makes the tie decision deterministic and stable.
    scaled_weights = sorted_weights / float(np.max(sorted_weights))
    total_weight = math.fsum(float(weight) for weight in scaled_weights)
    cumulative = 0.0
    weighted_median = float(sorted_ratios[-1])
    for ratio, weight in zip(sorted_ratios, scaled_weights):
        cumulative = math.fsum((cumulative, float(weight)))
        if 2.0 * cumulative >= total_weight:
            weighted_median = float(ratio)
            break
    return max(0.0, weighted_median)


__all__ = [
    "fit_model",
    "predict_model",
    "fit_tree_e1_backfit",
    "fit_ridge_e1_within",
    "predict_direct_antisymmetric",
    "nonnegative_l1_scale",
]
