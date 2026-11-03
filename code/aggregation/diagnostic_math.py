"""Pure formulas ported from accepted Route C diagnostics and post hoc accounting.

No model imports, fitted-object loading, prediction, or file I/O occurs here.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

TOL = 1e-12
GROUPS = ["persistent_zero", "restored", "both_nonzero"]
KEY = ["repeat", "outer_fold", "pair_id"]
COMBO = ["representation", "learner", "pipeline", "scale"]
POSTHOC = "post_hoc_R5_existing_diagnostic_update_20261009"


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    require(np.isfinite(y).all() and np.isfinite(p).all(), "nonfinite metric input")
    nz = y != 0
    zero = np.abs(p[nz]) <= TOL
    correct = (np.sign(y[nz]) == np.sign(p[nz])) & ~zero
    return {"n": len(y), "n_direction": int(nz.sum()),
            "mae": float(np.mean(np.abs(y-p))) if len(y) else None,
            "rmse": float(np.sqrt(np.mean((y-p)**2))) if len(y) else None,
            "DA_strict": float(correct.mean()) if nz.any() else None,
            "DA_half": float(np.mean(correct.astype(float) + .5*zero)) if nz.any() else None,
            "zero_predictions": int((np.abs(p) <= TOL).sum()),
            "true_ties_excluded": int((~nz).sum())}


def direction_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    active = y != 0.0
    signs = np.sign(p)
    signs[np.abs(p) <= 1e-12] = 0.0
    predicted_zero = signs[active] == 0.0
    correct = np.sign(y[active]) == signs[active]
    return {"DA_strict": float(np.mean(correct & ~predicted_zero)) if active.any() else None,
            "DA_half": float(np.mean(np.where(predicted_zero, .5, correct.astype(float)))) if active.any() else None,
            "zero_predictions": int(predicted_zero.sum()), "true_ties_excluded": int((~active).sum())}


def project_block(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], dict]:
    """Project edge p onto incidence gradients, independently in each component."""
    frame = frame.copy().reset_index(drop=True)
    nodes = sorted(set(frame.record_id_i.astype(str)) | set(frame.record_id_j.astype(str)))
    index = {n: i for i, n in enumerate(nodes)}
    pi = np.asarray([index[str(n)] for n in frame.record_id_i], int)
    pj = np.asarray([index[str(n)] for n in frame.record_id_j], int)
    adjacency = coo_matrix((np.ones(2*len(frame)), (np.r_[pi, pj], np.r_[pj, pi])),
                           shape=(len(nodes), len(nodes))).tocsr()
    n_components, labels = connected_components(adjacency, directed=False)
    potentials, projected = [], {col: np.zeros(len(frame)) for col in ("raw_prediction", "calibrated_prediction")}
    component_id = np.zeros(len(frame), int)
    for component in range(n_components):
        members = np.flatnonzero(labels == component)
        positions = np.flatnonzero(labels[pi] == component)
        if not np.all(labels[pj[positions]] == component):
            raise RuntimeError("edge crosses disconnected components")
        local = {node: i for i, node in enumerate(members)}
        B = np.zeros((len(positions), len(members)), float)
        B[np.arange(len(positions)), [local[p] for p in pi[positions]]] = -1
        B[np.arange(len(positions)), [local[p] for p in pj[positions]]] = 1
        component_id[positions] = component
        phi_values = {}
        for col in projected:
            p = frame[col].to_numpy(float)[positions]
            # The minimum-norm solution gives a centred gauge; labels are not read.
            phi = np.linalg.lstsq(B, p, rcond=None)[0]
            projected[col][positions] = B @ phi
            phi_values[col] = phi
        for node, local_index in local.items():
            potentials.append({"record_id": nodes[node], "component_id": component,
                "raw_projected_potential": phi_values["raw_prediction"][local_index],
                "calibrated_projected_potential": phi_values["calibrated_prediction"][local_index]})
    frame["component_id"] = component_id
    y = frame.true_delta.to_numpy(float)
    summary = {"records": len(nodes), "edges": len(frame), "components": n_components,
               "cycle_rank": len(frame)-len(nodes)+n_components}
    for col, value in projected.items():
        prefix = "raw" if col == "raw_prediction" else "calibrated"
        p = frame[col].to_numpy(float)
        residual = p - value
        numerator, denominator = float(np.dot(residual, residual)), float(np.dot(p, p))
        frame[f"{prefix}_projected_prediction"] = value
        frame[f"{prefix}_cycle_residual"] = residual
        summary.update({f"{prefix}_squared_cycle_l2": numerator, f"{prefix}_squared_prediction_l2": denominator,
            f"{prefix}_kappa": float(np.sqrt(numerator/denominator)) if denominator else 0.0,
            f"{prefix}_zero_denominator": denominator == 0,
            f"{prefix}_mae": float(np.mean(np.abs(y-p))),
            f"{prefix}_projected_mae": float(np.mean(np.abs(y-value))),
            f"{prefix}_rmse": float(np.sqrt(np.mean((y-p)**2))),
            f"{prefix}_projected_rmse": float(np.sqrt(np.mean((y-value)**2)))})
        summary.update({f"{prefix}_{key}": value for key, value in direction_metrics(y, p).items()})
        summary.update({f"{prefix}_projected_{key}": value for key, value in direction_metrics(y, projected[col]).items()})
    return frame, potentials, summary


def signature(counts: dict[int, int]) -> str:
    payload = json.dumps(sorted(counts.items()), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest().upper()


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


def digest(X):
    # +0 and -0 are the same predictor input. Canonicalize the zero sign before
    # hashing reverse inputs, rather than spuriously separating signed zeros.
    a = np.asarray(X, dtype=np.float32).copy()
    a[a == 0] = 0.0
    return hashlib.sha256(a.tobytes()).hexdigest().upper()


def pair_inputs(bundle, X, representation, layer, population):
    rows = []
    for pair, i, j in zip(bundle.pairs.itertuples(index=False), bundle.pair_i, bundle.pair_j):
        d, s = X[j] - X[i], X[j] + X[i]
        zero = bool(np.all(d == 0))
        dh, dr = digest(d), digest(-d)
        ch, cr = digest(np.r_[d, s]), digest(np.r_[-d, s])
        orbit_d, orbit_c = min(dh, dr), min(ch, cr)
        rows.append({"representation": representation, "layer": layer, "population": population,
            "pair_id": pair.pair_id, "record_id_i": pair.record_id_i, "record_id_j": pair.record_id_j,
            "direct_d_signature": dh, "direct_reverse_signature": dr,
            "direct_signed_orbit_id": orbit_d, "direct_orbit_orientation": 0 if zero else (1 if dh == orbit_d else -1),
            "C1_ds_signature": ch, "C1_reverse_signature": cr,
            "C1_signed_orbit_id": orbit_c, "C1_orbit_orientation": 0 if zero else (1 if ch == orbit_c else -1),
            "exact_zero_d": zero})
    result = pd.DataFrame(rows)
    if result.pair_id.duplicated().any():
        raise RuntimeError("pair input diagnostic keys not unique")
    return result


def pair_input_census(table, *, member_layer, repeat, outer_fold, source_path):
    rows = []
    for arm, feature, orbit in (("direct_pair", "direct_d_signature", "direct_signed_orbit_id"),
                                ("context_pair_C1", "C1_ds_signature", "C1_signed_orbit_id")):
        classes, orbits = table.groupby(feature).size(), table.groupby(orbit).size()
        rows.append({"representation": table.representation.iloc[0], "layer": member_layer,
            "repeat": repeat, "outer_fold": outer_fold, "population": table.population.iloc[0], "pipeline": arm,
            "pairs": len(table), "canonical_input_classes": len(classes), "signed_input_orbits": len(orbits),
            "canonical_duplicated_classes": int((classes > 1).sum()),
            "pairs_in_canonical_duplicate_classes": int(classes[classes > 1].sum()),
            "signed_repeated_orbits": int((orbits > 1).sum()),
            "pairs_in_signed_repeated_orbits": int(orbits[orbits > 1].sum()),
            "exact_zero_difference_pairs": int(table.exact_zero_d.sum()),
            "source_path": source_path.as_posix()})
    return rows


def classify(binary, count):
    binary, count = np.asarray(binary, bool), np.asarray(count, bool)
    return np.select([binary & count, binary & ~count, ~binary & ~count],
                     GROUPS, default="count_only_zero")


def census(frame, **meta):
    identities = pd.unique(pd.concat([frame.identity_hash_i, frame.identity_hash_j], ignore_index=True))
    records = pd.unique(pd.concat([frame.record_id_i, frame.record_id_j], ignore_index=True))
    row = {"analysis_status": POSTHOC, **meta, "observations": len(frame),
           "unique_edges": frame.pair_id.nunique(), "chemical_identities": len(identities),
           "records": len(records), "condition_blocks": frame[["normalized_poi_id", "pair_context_hash"]].drop_duplicates().shape[0]}
    if "dependency_component" in frame:
        row["accepted_internal_dependency_components"] = frame.dependency_component.nunique()
        row["external_series_components"] = None
        row["dependency_definition"] = "R5 frozen source-plus-shared-identity dependency mapping; counts are overlapping descriptive support, not effective N"
    else:
        row["accepted_internal_dependency_components"] = None
        row["external_series_components"] = int(len(frame) > 0)
        row["dependency_definition"] = "single previously seen source/series with shared endpoints; no accepted internal DEP id applies"
    return row


def census_tables(mapping, scope):
    rows = []
    for group in GROUPS:
        rows.append(census(mapping.loc[mapping.group.eq(group)], scope=scope, repeat=-1, outer_fold=-1,
                           group=group, aggregation="pooled observation support; unique counts deduplicated"))
    if mapping.repeat.ge(0).any():
        for repeat, rep in mapping.groupby("repeat", sort=True):
            for group in GROUPS:
                rows.append(census(rep.loc[rep.group.eq(group)], scope=scope, repeat=int(repeat), outer_fold=-1,
                                   group=group, aggregation="within-repeat support"))
            for fold, cell in rep.groupby("outer_fold", sort=True):
                for group in GROUPS:
                    rows.append(census(cell.loc[cell.group.eq(group)], scope=scope, repeat=int(repeat), outer_fold=int(fold),
                                       group=group, aggregation="actual outer-fold support"))
    return pd.DataFrame(rows)


def attach(pred, mapping, scope):
    cols = KEY + ["group", "binary_exact_zero_d", "count_exact_zero_d"]
    if scope == "external_ensemble":
        cols = ["pair_id", "group", "binary_exact_zero_d", "count_exact_zero_d"]
        merge = ["pair_id"]
    else:
        merge = KEY
    frame = pred.merge(mapping[cols], on=merge, validate="m:1")
    require(len(frame) == len(pred), "input mapping changed prediction support")
    frame["structural_zero_d"] = np.where(frame.representation.eq("binary"),
                                          frame.binary_exact_zero_d, frame.count_exact_zero_d)
    frame["raw_zero_tol"] = frame.raw_prediction.abs().le(TOL)
    frame["calibrated_zero_tol"] = frame.calibrated_prediction.abs().le(TOL)
    require(frame.loc[frame.structural_zero_d, "raw_zero_tol"].all(), "structural same-input raw zero violated")
    require(frame.loc[frame.structural_zero_d, "calibrated_zero_tol"].all(), "structural same-input calibrated zero violated")
    return frame


def metric_row(frame, total_n, meta, scale):
    col = "raw_prediction" if scale == "raw" else "calibrated_prediction"
    n = len(frame)
    y, p = frame.true_delta.to_numpy(float), frame[col].to_numpy(float)
    values = metrics(y, p)
    loss_sum, zero_sum = float(np.abs(y-p).sum()), float(np.abs(y).sum())
    zero_mae = zero_sum/n if n else np.nan
    values.update({"analysis_status": POSTHOC, **meta, "scale": scale, "total_observations_in_scope": total_n,
                   "absolute_loss_sum": loss_sum, "zero_absolute_loss_sum": zero_sum,
                   "own_group_zero_mae": zero_mae,
                   "mae_reduction_vs_own_zero": 1-loss_sum/zero_sum if zero_sum else np.nan,
                   "loss_contribution_to_overall_mae": loss_sum/total_n if total_n else np.nan,
                   "zero_loss_contribution_to_overall_mae": zero_sum/total_n if total_n else np.nan,
                   "correct_direction_count": int(((np.sign(y) == np.sign(p)) & (np.abs(p) > TOL) & (y != 0)).sum()),
                   "zero_direction_predictions": int(((np.abs(p) <= TOL) & (y != 0)).sum()),
                   "exact_zero_predictions": int((p == 0).sum()),
                   "structural_zero_d_observations": int(frame.structural_zero_d.sum()),
                   "nonstructural_zero_predictions_tol": int((~frame.structural_zero_d & (frame[col].abs() <= TOL)).sum())})
    return values


def grouped_metrics(frame, scope, by):
    rows = []
    base = ["representation", "learner", "pipeline"] + by
    for keys, cell in frame.groupby(base, sort=True):
        meta = {"scope": scope, **dict(zip(base, keys))}
        for group in ["all"] + GROUPS:
            sub = cell if group == "all" else cell.loc[cell.group.eq(group)]
            for scale in ["raw", "calibrated"]:
                rows.append(metric_row(sub, len(cell), {**meta, "group": group}, scale))
    table = pd.DataFrame(rows)
    # Sum group loss, denominators and direction counts back to their own scope's all row.
    errs = []
    for keys, cell in table.groupby(base + ["scale"], sort=True):
        whole = cell.loc[cell.group.eq("all")].iloc[0]
        groups = cell.loc[cell.group.isin(GROUPS)]
        require(int(groups.n.sum()) == int(whole.n), "group observation counts do not close")
        for col in ["absolute_loss_sum", "zero_absolute_loss_sum", "n_direction", "zero_predictions",
                    "correct_direction_count", "zero_direction_predictions", "exact_zero_predictions"]:
            err = float(groups[col].sum()-whole[col])
            require(abs(err) <= 2e-10, f"group {col} does not close")
            errs.append({"scope": scope, **dict(zip(base + ["scale"], keys)), "quantity": col,
                         "group_sum": float(groups[col].sum()), "overall_value": float(whole[col]), "residual": err})
        err = float(groups.loss_contribution_to_overall_mae.sum()-whole.mae)
        require(abs(err) <= TOL, "group weighted MAE contributions do not close")
        errs.append({"scope": scope, **dict(zip(base + ["scale"], keys)), "quantity": "mae_contribution",
                     "group_sum": float(groups.loss_contribution_to_overall_mae.sum()), "overall_value": float(whole.mae),
                     "residual": err})
    return table, pd.DataFrame(errs)


def repeat_equal_summary(table):
    keys = ["scope"] + COMBO + ["group"]
    rows = []
    for values, frame in table.groupby(keys, sort=True):
        row = {"analysis_status": POSTHOC, **dict(zip(keys, values)), "repeats_total": len(frame),
               "repeats_nonempty": int(frame.n.gt(0).sum()), "observations_across_repeats": int(frame.n.sum())}
        for col in ["n", "mae", "own_group_zero_mae", "mae_reduction_vs_own_zero", "DA_strict", "DA_half",
                    "zero_predictions", "exact_zero_predictions", "nonstructural_zero_predictions_tol",
                    "loss_contribution_to_overall_mae", "zero_loss_contribution_to_overall_mae"]:
            for stat in ["mean", "min", "max"]:
                row[f"{stat}_{col}"] = float(getattr(frame[col], stat)()) if frame[col].notna().any() else np.nan
        row["ratio_of_repeatmean_mae_reduction_vs_zero"] = (
            1-row["mean_mae"]/row["mean_own_group_zero_mae"] if row["mean_own_group_zero_mae"] else np.nan)
        row["aggregation"] = "within-repeat group loss pooled over actual five-fold observations; three repeats equally weighted"
        rows.append(row)
    return pd.DataFrame(rows)


def representation_differences(table, repeat=True):
    keys = ["scope", "learner", "pipeline", "scale", "group"] + (["repeat"] if repeat else [])
    b = table.loc[table.representation.eq("binary")].set_index(keys)
    c = table.loc[table.representation.eq("count")].set_index(keys)
    require(b.index.equals(c.index), "representation group metric keys differ")
    require(b.n.eq(c.n).all(), "representation group metric supports differ")
    out = b[["n", "total_observations_in_scope"]].copy()
    out["mae_binary"], out["mae_count"] = b.mae, c.mae
    out["mae_difference_count_minus_binary"] = c.mae-b.mae
    out["absolute_loss_difference_sum"] = c.absolute_loss_sum-b.absolute_loss_sum
    out["difference_contribution_to_overall_mae"] = c.loss_contribution_to_overall_mae-b.loss_contribution_to_overall_mae
    out["analysis_status"] = POSTHOC
    out = out.reset_index()
    closure = []
    checkkeys = ["scope", "learner", "pipeline", "scale"] + (["repeat"] if repeat else [])
    for values, frame in out.groupby(checkkeys):
        allrow = frame.loc[frame.group.eq("all")].iloc[0]
        delta = frame.loc[frame.group.isin(GROUPS), "difference_contribution_to_overall_mae"].sum()
        err = float(delta-allrow.mae_difference_count_minus_binary)
        require(abs(err) <= TOL, "representation group contributions fail overall loss-difference closure")
        closure.append({"scope": values[0], **dict(zip(checkkeys, values)), "group_sum": float(delta),
                        "overall_mae_difference": float(allrow.mae_difference_count_minus_binary), "residual": err})
    return out, pd.DataFrame(closure)


def difference_mean(table):
    keys = ["scope", "learner", "pipeline", "scale", "group"]
    numeric = ["n", "mae_binary", "mae_count", "mae_difference_count_minus_binary",
               "absolute_loss_difference_sum", "difference_contribution_to_overall_mae"]
    rows = []
    for values, frame in table.groupby(keys, sort=True):
        row = {"analysis_status": POSTHOC, **dict(zip(keys, values)), "observations_across_repeats": int(frame.n.sum())}
        for col in numeric:
            for stat in ["mean", "min", "max"]:
                row[f"{stat}_{col}"] = float(getattr(frame[col], stat)()) if frame[col].notna().any() else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    for _, frame in out.groupby(["scope", "learner", "pipeline", "scale"]):
        require(abs(frame.loc[frame.group.isin(GROUPS), "mean_difference_contribution_to_overall_mae"].sum()-
                    frame.loc[frame.group.eq("all"), "mean_mae_difference_count_minus_binary"].iloc[0]) <= TOL,
                "repeat equal difference contribution closure fails")
    return out


def zero_evidence(frame, scope, by):
    rows = []
    keys = ["representation", "learner", "pipeline"] + by + ["group"]
    for values, cell in frame.groupby(keys, sort=True):
        forced = cell.structural_zero_d
        rawzero, calzero = cell.raw_zero_tol, cell.calibrated_zero_tol
        row = {"analysis_status": POSTHOC, "scope": scope, **dict(zip(keys, values)), "n": len(cell),
               "structural_zero_d": int(forced.sum()), "raw_zero_tol": int(rawzero.sum()),
               "raw_exact_zero": int(cell.raw_prediction.eq(0).sum()), "calibrated_zero_tol": int(calzero.sum()),
               "calibrated_exact_zero": int(cell.calibrated_prediction.eq(0).sum()),
               "nonstructural_raw_zero_origin_unidentified": int((~forced & rawzero).sum())}
        if "scale_c" in cell:
            c0 = cell.scale_c.eq(0)
            row.update({"scale_c_zero": int(c0.sum()),
                        "nonstructural_cal_zero_supported_by_c0_nonzero_raw": int((~forced & calzero & ~rawzero & c0).sum()),
                        "nonstructural_cal_zero_raw_nonzero_c_positive_tol_only": int((~forced & calzero & ~rawzero & ~c0).sum())})
        rows.append(row)
    return pd.DataFrame(rows)
