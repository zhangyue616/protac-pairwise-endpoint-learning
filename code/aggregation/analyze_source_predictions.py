"""Frozen finite-population summaries for the accepted R6 protocol.

Consumes a consolidated table only after complete, authorized production.
No model fitting, prediction, resampling, subgroup search, or p-values.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

ARMS = ("direct_pair", "endpoint_E0", "endpoint_E0_matched", "endpoint_E1")
LEARNERS = ("ridge", "random_forest", "xgboost")
REPRESENTATIONS = ("binary", "count")
TIE_TOLERANCE = 1e-12


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def rho(frame, outcome="delta_mae"):
    x = frame.primary_support_median_identity_nn.rank(method="average").to_numpy(float)
    y = frame[outcome].rank(method="average").to_numpy(float)
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def summary(frame, representation, scope, omitted=None):
    return {"representation": representation, "scope": scope,
            "omitted_component": omitted, "components": len(frame),
            "theta_E1_minus_E0matched": float(frame.delta_mae.mean()),
            "spearman_support_delta": rho(frame),
            "groups_E1_better": int((frame.delta_mae < -TIE_TOLERANCE).sum()),
            "groups_E0matched_better": int((frame.delta_mae > TIE_TOLERANCE).sum()),
            "groups_tied": int((frame.delta_mae.abs() <= TIE_TOLERANCE).sum())}


def analyze(predictions, pair_map, census, support):
    needed = {"heldout_component_id", "representation", "learner", "pipeline",
              "pair_id", "true_delta", "raw_prediction", "calibrated_prediction"}
    require(needed <= set(predictions.columns), "Prediction columns missing")
    require(len(pair_map) == 5788 and pair_map.pair_id.is_unique, "Pair-map census drift")
    selected = census.loc[census.pairs > 0].copy()
    require(len(selected) == 43 and selected.component_id.is_unique, "Qualified-group census drift")
    groups = set(selected.component_id)
    require(set(support.dependency_component) == groups and support.dependency_component.is_unique,
            "Structural support does not cover exactly the selected groups")
    require(set(pair_map.heldout_component_id) == groups, "Pair map group population drift")
    expected_pair_group = pair_map.set_index("pair_id").heldout_component_id
    require(set(predictions.pair_id) == set(expected_pair_group.index), "Prediction pair population drift")
    require((predictions.pair_id.map(expected_pair_group) == predictions.heldout_component_id).all(),
            "Prediction component differs from pair map")
    require(set(predictions.representation) == set(REPRESENTATIONS), "Representation scope drift")
    require(set(predictions.learner) == set(LEARNERS), "Learner scope drift")
    require(set(predictions.pipeline) == set(ARMS), "Strategy scope drift")
    key = ["representation", "learner", "pipeline", "pair_id"]
    require(not predictions.duplicated(key).any(), "Duplicate held-out prediction")
    require(len(predictions) == 5788 * 2 * 3 * 4, "Incomplete prediction grid")
    for _, part in predictions.groupby(key[:-1], sort=True):
        require(set(part.pair_id) == set(expected_pair_group.index), "Incomplete strategy test population")
    values = predictions[["true_delta", "calibrated_prediction"]].to_numpy(float)
    require(np.isfinite(values).all(), "Nonfinite prediction/target")
    require(np.isfinite(predictions.raw_prediction.to_numpy(float)).all(), "Nonfinite raw prediction")
    require(predictions.groupby("pair_id").true_delta.nunique().eq(1).all(), "Target changes across arms")
    data = predictions.copy()
    data["absolute_error"] = np.abs(values[:, 0] - values[:, 1])
    data["squared_error"] = np.square(values[:, 0] - values[:, 1])
    data["raw_absolute_error"] = np.abs(data.true_delta - data.raw_prediction)
    data["raw_squared_error"] = np.square(data.true_delta - data.raw_prediction)
    columns = ["heldout_component_id", "representation", "learner", "pipeline"]
    metrics = data.groupby(columns, sort=True).agg(
        test_pairs=("pair_id", "size"), mae=("absolute_error", "mean"),
        mse=("squared_error", "mean"), raw_mae=("raw_absolute_error", "mean"),
        raw_mse=("raw_squared_error", "mean")).reset_index()
    metrics["rmse"] = np.sqrt(metrics.pop("mse"))
    metrics["raw_rmse"] = np.sqrt(metrics.pop("raw_mse"))
    require(len(metrics) == 43 * 24, "Group/strategy grid incomplete")
    true_once = data[["heldout_component_id", "pair_id", "true_delta"]].drop_duplicates("pair_id")
    true_once["zero_absolute_error"] = true_once.true_delta.abs()
    true_once["zero_squared_error"] = true_once.true_delta.pow(2)
    zero = true_once.groupby("heldout_component_id", sort=True).agg(
        test_pairs=("pair_id", "size"), zero_mae=("zero_absolute_error", "mean"),
        zero_mse=("zero_squared_error", "mean")).reset_index()
    zero["zero_rmse"] = np.sqrt(zero.pop("zero_mse"))
    metrics = metrics.merge(zero[["heldout_component_id", "zero_mae", "zero_rmse"]],
                            on="heldout_component_id", validate="many_to_one")
    metrics["mae_minus_zero"] = metrics.mae - metrics.zero_mae
    macro = metrics.groupby(["representation", "learner", "pipeline"], sort=True).agg(
        group_macro_mae=("mae", "mean"), group_macro_rmse=("rmse", "mean"),
        group_macro_mae_minus_zero=("mae_minus_zero", "mean"),
        raw_group_macro_mae=("raw_mae", "mean"),
        raw_group_macro_rmse=("raw_rmse", "mean")).reset_index()
    pooled = data.groupby(["representation", "learner", "pipeline"], sort=True).agg(
        test_pairs=("pair_id", "size"), pair_pooled_mae=("absolute_error", "mean"),
        pair_pooled_mse=("squared_error", "mean"),
        raw_pair_pooled_mae=("raw_absolute_error", "mean"),
        raw_pair_pooled_mse=("raw_squared_error", "mean")).reset_index()
    pooled["pair_pooled_rmse"] = np.sqrt(pooled.pop("pair_pooled_mse"))
    pooled["raw_pair_pooled_rmse"] = np.sqrt(pooled.pop("raw_pair_pooled_mse"))
    pivot = metrics.pivot(index=["heldout_component_id", "representation", "learner"], columns="pipeline", values="mae")
    contrast = (pivot.endpoint_E1 - pivot.endpoint_E0_matched).rename("delta_mae").reset_index()
    average = contrast.groupby(["heldout_component_id", "representation"], sort=True).delta_mae.mean().reset_index()
    arm_average = pivot.groupby(level=["heldout_component_id", "representation"], sort=True).mean()
    arm_average = arm_average[["endpoint_E1", "endpoint_E0_matched"]].rename(
        columns={"endpoint_E1": "A_E1_learner_mean_mae", "endpoint_E0_matched": "B_E0matched_learner_mean_mae"}).reset_index()
    average = average.merge(arm_average, on=["heldout_component_id", "representation"], validate="one_to_one")
    denominator = average.A_E1_learner_mean_mae + average.B_E0matched_learner_mean_mae
    require((denominator >= 0).all() and np.isfinite(denominator).all(), "Invalid bounded-contrast denominator")
    average["bounded_relative_difference_u"] = np.divide(
        average.A_E1_learner_mean_mae - average.B_E0matched_learner_mean_mae, denominator,
        out=np.zeros(len(average), dtype=float), where=denominator.to_numpy() != 0.0)
    require(average.bounded_relative_difference_u.between(-1, 1).all(), "u outside its declared bounds")
    metadata = selected[["component_id", "records", "identities", "pairs", "pair_endpoint_identities"]]
    average = average.merge(metadata, left_on="heldout_component_id", right_on="component_id", validate="many_to_one")
    average = average.merge(support[["dependency_component", "primary_support_median_identity_nn",
                                     "matched_train_records", "matched_train_identities"]],
                            left_on="heldout_component_id", right_on="dependency_component", validate="many_to_one")
    average = average.merge(zero[["heldout_component_id", "zero_mae", "zero_rmse"]],
                            on="heldout_component_id", validate="many_to_one")
    contrast = contrast.merge(average[["heldout_component_id", "representation", "primary_support_median_identity_nn"]],
                              on=["heldout_component_id", "representation"], validate="many_to_one")
    largest_two = tuple(selected.sort_values(["pairs", "component_id"], ascending=[False, True]).component_id.iloc[:2])
    require(largest_two == ("DEP_006", "DEP_002"), "Prespecified largest groups changed")
    require(int(selected.pair_endpoint_identities.ge(5).sum()) == 25, "Prespecified sensitivity population changed")
    summaries, sensitivity, scale_diagnostic = [], [], []
    for representation in REPRESENTATIONS:
        frame = average.loc[average.representation.eq(representation)]
        summaries.append(summary(frame, representation, "all_43_components"))
        scale_diagnostic.append({"representation": representation, "scope": "all_43_components_only",
                                 "components": len(frame),
                                 "spearman_support_bounded_relative_difference": rho(frame, "bounded_relative_difference_u"),
                                 "definition": "u=(mean_learner_MAE_E1-mean_learner_MAE_E0matched)/(sum); exactly zero sum maps to zero",
                                 "scope_limit": "Scale diagnostic; no confounder adjustment or model-choice validation"})
        sensitivity.append(summary(frame.loc[frame.pair_endpoint_identities.ge(5)], representation,
                                   "at_least_5_pair_endpoint_identities"))
        sensitivity.append(summary(frame.loc[~frame.heldout_component_id.isin(largest_two)], representation,
                                   "omit_two_largest_pair_groups", ";".join(largest_two)))
        for group in sorted(groups):
            sensitivity.append(summary(frame.loc[frame.heldout_component_id.ne(group)], representation,
                                       "delete_one_component_fixed_predictions", group))
    learner_summary = contrast.groupby(["representation", "learner"], sort=True).delta_mae.agg(["mean", "min", "max"]).reset_index()
    learner_summary = learner_summary.rename(columns={"mean": "theta_E1_minus_E0matched", "min": "smallest_group_delta", "max": "largest_group_delta"})
    return {"GROUP_METRICS": metrics, "ZERO_BY_GROUP": zero, "MACRO_METRICS": macro,
            "POOLED_METRICS": pooled, "GROUP_LEARNER_CONTRASTS": contrast,
            "GROUP_AVERAGE_CONTRASTS": average, "LEARNER_CONTRAST_SUMMARY": learner_summary,
            "PRIMARY_SUMMARY": pd.DataFrame(summaries), "SENSITIVITY": pd.DataFrame(sensitivity),
            "SCALE_DIAGNOSTIC_ALL_43": pd.DataFrame(scale_diagnostic)}


def main():
    parser = argparse.ArgumentParser()
    for option in ("predictions", "pair-map", "census", "support", "completion-receipt", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.completion_receipt.read_text(encoding="utf-8"))
    require(receipt.get("status") == "COMPLETE_R6_PRODUCTION" and receipt.get("completed_jobs") == 1032,
            "Analysis requires complete authorized R6 production")
    require(receipt.get("scientific_execution_authorized") is True and bool(receipt.get("freeze_sha256")),
            "Missing authorized frozen run binding")
    result = analyze(pd.read_csv(args.predictions), pd.read_csv(args.pair_map),
                     pd.read_csv(args.census), pd.read_csv(args.support))
    require(not args.output.exists(), "Use a fresh analysis output directory")
    args.output.mkdir(parents=True)
    for name, table in result.items():
        table.to_csv(args.output / (name + ".csv"), index=False, lineterminator="\n")
    scope = {"status": "FINITE_POPULATION_DESCRIPTIVE_SUMMARY", "groups": 43,
             "pair_rows_per_strategy": 5788, "final_strategy_slots": 1032,
             "additional_fits": 0, "additional_predictions": 0, "p_values": False,
             "confidence_intervals": False, "family_independence_verified": False,
             "scale_diagnostic_full_43_only": True, "raw_and_calibrated_metrics_saved": True,
             "interpretation": "Fixed source/shared-identity groups; overlapping training; no validated model-choice rule; no total training uncertainty estimate.",
             "completion_receipt": str(args.completion_receipt.resolve())}
    (args.output / "ANALYSIS_SCOPE.json").write_text(json.dumps(scope, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(scope))


if __name__ == "__main__":
    main()
