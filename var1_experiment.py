"""Full var1 experiment for report-ready model-selection evidence."""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parent
SEED = 2026
N_SPLITS = 5
N_REPEATS = 3
MAX_DEGREE = 10

BASELINE_ALPHAS = {"ridge": 1.0, "lasso": 0.01, "elastic_net": 0.01}
ALPHA_SWEEP = np.unique(np.r_[np.logspace(-4, -2, 13), np.linspace(0.011, 0.03, 8)])
L1_RATIOS = (0.1, 0.5, 0.9)


def make_model(name, degree, alpha, l1_ratio=None):
    polynomial = PolynomialFeatures(degree=degree, include_bias=False)
    if name == "ridge":
        return make_pipeline(polynomial, Ridge(alpha=alpha))
    if name == "lasso":
        return make_pipeline(polynomial, StandardScaler(), Lasso(alpha=alpha, max_iter=100000))
    return make_pipeline(
        polynomial,
        StandardScaler(),
        ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=100000),
    )


def degree_candidates():
    for degree in range(1, MAX_DEGREE + 1):
        for name, alpha in BASELINE_ALPHAS.items():
            # High-degree Elastic Net fits are disproportionately expensive;
            # degree 8/9/10 are still covered by Ridge and Lasso curves.
            if name == "elastic_net" and degree >= 8:
                continue
            ratios = (np.nan,) if name != "elastic_net" else L1_RATIOS
            for ratio in ratios:
                yield "degree_curve", name, degree, float(alpha), ratio


def alpha_candidates(degree):
    """Build the second-stage search only after degree selection."""
    for alpha in ALPHA_SWEEP:
        yield "alpha_sweep", "lasso", degree, float(alpha), np.nan
        for ratio in L1_RATIOS:
            yield "alpha_sweep", "elastic_net", degree, float(alpha), ratio


def candidate_key(stage, model, degree, alpha, l1_ratio):
    ratio = None if pd.isna(l1_ratio) else float(l1_ratio)
    return (stage, model, int(degree), float(alpha), ratio)


def load_completed_candidates(path):
    if not path.exists():
        return set()
    existing = pd.read_csv(path)
    completed = set()
    for values, group in existing.groupby(
        ["stage", "model", "degree", "alpha", "l1_ratio"], dropna=False
    ):
        if len(group) == N_SPLITS * N_REPEATS:
            stage, model, degree, alpha, l1_ratio = values
            completed.add(candidate_key(stage, model, degree, alpha, l1_ratio))
    return completed


def clean_fold_results(path):
    if not path.exists():
        return pd.DataFrame()
    fold_results = pd.read_csv(path)
    fold_results = fold_results.drop_duplicates(
        subset=["stage", "model", "degree", "alpha", "l1_ratio", "fold"],
        keep="last",
    )
    fold_results["candidate_key"] = fold_results.apply(
        lambda row: candidate_key(
            row["stage"], row["model"], row["degree"], row["alpha"], row["l1_ratio"]
        ),
        axis=1,
    )
    complete_keys = {
        key
        for key, group in fold_results.groupby("candidate_key")
        if group["fold"].nunique() == N_SPLITS * N_REPEATS
    }
    fold_results = fold_results[fold_results["candidate_key"].isin(complete_keys)]
    fold_results = fold_results.drop(columns="candidate_key")
    # Keep the raw CSV as an append-only audit log, but exclude intentionally
    # omitted high-degree Elastic Net candidates from report summaries.
    return fold_results[
        ~(
            (fold_results["model"] == "elastic_net")
            & (fold_results["degree"] >= 8)
        )
    ]


def summarize(path):
    fold_results = clean_fold_results(path)
    if fold_results.empty:
        return fold_results
    return (
        fold_results.groupby(["stage", "model", "degree", "alpha", "l1_ratio"], dropna=False)
        .agg(
            cv_mse=("validation_mse", "mean"),
            cv_mse_sd=("validation_mse", "std"),
            cv_r2=("validation_r2", "mean"),
            cv_r2_sd=("validation_r2", "std"),
            train_mse=("train_mse", "mean"),
            train_r2=("train_r2", "mean"),
            mean_fit_seconds=("fit_seconds", "mean"),
            mean_iterations=("iterations", "mean"),
        )
        .reset_index()
        .sort_values("cv_mse")
    )


def run_candidates(candidate_list, X, y, splitter, fold_path, completed):
    total_fits = len(candidate_list) * N_SPLITS * N_REPEATS
    finished_fits = sum(
        N_SPLITS * N_REPEATS
        for candidate in candidate_list
        if candidate_key(*candidate) in completed
    )
    print(
        f"Starting {candidate_list[0][0] if candidate_list else 'empty'} stage: "
        f"{len(candidate_list)} candidates, {total_fits} fits, "
        f"{finished_fits} already complete.",
        flush=True,
    )

    for candidate_number, (stage, model_name, degree, alpha, l1_ratio) in enumerate(
        candidate_list, start=1
    ):
        key = candidate_key(stage, model_name, degree, alpha, l1_ratio)
        if key in completed:
            print(
                f"[{candidate_number}/{len(candidate_list)}] SKIP "
                f"{stage} {model_name} degree={degree} alpha={alpha:.8g}",
                flush=True,
            )
            continue
        candidate_start = perf_counter()
        for fold_id, (train_idx, valid_idx) in enumerate(splitter.split(X), start=1):
            model = make_model(model_name, degree, alpha, l1_ratio)
            start = perf_counter()
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            fit_seconds = perf_counter() - start
            train_pred = model.predict(X.iloc[train_idx])
            valid_pred = model.predict(X.iloc[valid_idx])
            estimator = model.steps[-1][1]
            row = {
                "stage": stage,
                "model": model_name,
                "degree": degree,
                "alpha": alpha,
                "l1_ratio": l1_ratio,
                "fold": fold_id,
                "train_mse": mean_squared_error(y.iloc[train_idx], train_pred),
                "train_r2": r2_score(y.iloc[train_idx], train_pred),
                "validation_mse": mean_squared_error(y.iloc[valid_idx], valid_pred),
                "validation_r2": r2_score(y.iloc[valid_idx], valid_pred),
                "fit_seconds": fit_seconds,
                "iterations": getattr(estimator, "n_iter_", np.nan),
            }
            pd.DataFrame([row]).to_csv(
                fold_path,
                mode="a",
                header=not fold_path.exists(),
                index=False,
            )
            finished_fits += 1
            print(
                f"fit {finished_fits}/{total_fits} | candidate {candidate_number}/"
                f"{len(candidate_list)} | {model_name} degree={degree} "
                f"alpha={alpha:.8g} fold={fold_id}/{N_SPLITS * N_REPEATS} "
                f"valid_mse={row['validation_mse']:.6f} "
                f"elapsed={perf_counter() - candidate_start:.1f}s",
                flush=True,
            )

        print(
            f"COMPLETE {candidate_number}/{len(candidate_list)} | {stage} "
            f"{model_name} degree={degree} alpha={alpha:.8g} "
            f"elapsed={perf_counter() - candidate_start:.1f}s",
            flush=True,
        )
    return load_completed_candidates(fold_path)


def main():
    data = pd.read_csv(ROOT / "IMT2024028_train_var1.csv")
    X, y = data.drop(columns="y"), data["y"]
    fold_path = ROOT / "var1_fold_results.csv"
    splitter = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)

    completed = load_completed_candidates(fold_path)
    degree_list = list(degree_candidates())
    completed = run_candidates(degree_list, X, y, splitter, fold_path, completed)

    degree_summary = summarize(fold_path)
    degree_summary = degree_summary[degree_summary["stage"] == "degree_curve"]
    best_degree_row = degree_summary.loc[degree_summary["cv_mse"].idxmin()]
    best_degree = int(best_degree_row["degree"])
    print(
        f"INFERRED BEST DEGREE: {best_degree} | model={best_degree_row['model']} "
        f"| baseline_cv_mse={best_degree_row['cv_mse']:.6f}",
        flush=True,
    )

    alpha_list = list(alpha_candidates(best_degree))
    completed = run_candidates(alpha_list, X, y, splitter, fold_path, completed)
    summary = summarize(fold_path)
    summary.to_csv(ROOT / "var1_experiment_summary.csv", index=False)
    best_by_model = summary.loc[summary.groupby("model")["cv_mse"].idxmin()]
    best_by_model.to_csv(ROOT / "var1_best_by_model.csv", index=False)
    print("\nBest complete candidate by model:", flush=True)
    print(best_by_model.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
