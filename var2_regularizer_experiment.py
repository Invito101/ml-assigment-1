"""Unbiased var2 degree comparison across Ridge, Lasso, and Elastic Net.

Every degree and regularizer is evaluated with the same repeated folds. Results
are appended immediately and the run can resume without deleting prior rows.
"""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parent
SEED = 2026
N_SPLITS = 5
N_REPEATS = 3
FOLDS = N_SPLITS * N_REPEATS
MAX_DEGREE = 20

BASELINES = (
    ("ridge", 1e-8, np.nan),
    ("lasso", 0.01, np.nan),
    ("elastic_net", 0.01, 0.1),
    ("elastic_net", 0.01, 0.5),
    ("elastic_net", 0.01, 0.9),
)


def make_model(name, degree, alpha, l1_ratio):
    polynomial = PolynomialFeatures(degree, include_bias=False)
    if name == "ridge":
        return make_pipeline(polynomial, Ridge(alpha=alpha))
    if name == "lasso":
        return make_pipeline(polynomial, StandardScaler(), Lasso(alpha=alpha, max_iter=100000))
    return make_pipeline(
        polynomial,
        StandardScaler(),
        ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=100000),
    )


def candidate_key(name, degree, alpha, ratio):
    ratio = "none" if pd.isna(ratio) else f"{ratio:.8g}"
    return f"{name}|degree={degree}|alpha={alpha:.12g}|l1={ratio}"


def candidates():
    for degree in range(1, MAX_DEGREE + 1):
        for name, alpha, ratio in BASELINES:
            yield name, degree, alpha, ratio


def completed(path):
    if not path.exists():
        return set()
    data = pd.read_csv(path)
    counts = data.groupby("candidate_key")["fold"].nunique()
    return set(counts[counts == FOLDS].index)


def main():
    train = pd.read_csv(ROOT / "IMT2024028_train_var2.csv")
    X, y = train.drop(columns="y"), train["y"]
    path = ROOT / "var2_regularizer_fold_results.csv"
    split = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    all_candidates = list(candidates())
    done = completed(path)
    pending = [c for c in all_candidates if candidate_key(*c) not in done]
    print(
        f"Unbiased var2 degree comparison: {len(all_candidates)} candidates, "
        f"{len(pending)} pending, {len(pending) * FOLDS} pending fits.",
        flush=True,
    )

    for number, (name, degree, alpha, ratio) in enumerate(pending, start=1):
        start_candidate = perf_counter()
        for fold, (train_idx, valid_idx) in enumerate(split.split(X), start=1):
            model = make_model(name, degree, alpha, ratio)
            start_fit = perf_counter()
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            fit_seconds = perf_counter() - start_fit
            train_pred = model.predict(X.iloc[train_idx])
            valid_pred = model.predict(X.iloc[valid_idx])
            estimator = model.steps[-1][1]
            row = {
                "candidate_key": candidate_key(name, degree, alpha, ratio),
                "model": name,
                "degree": degree,
                "alpha": alpha,
                "l1_ratio": ratio,
                "fold": fold,
                "train_mse": mean_squared_error(y.iloc[train_idx], train_pred),
                "train_r2": r2_score(y.iloc[train_idx], train_pred),
                "validation_mse": mean_squared_error(y.iloc[valid_idx], valid_pred),
                "validation_r2": r2_score(y.iloc[valid_idx], valid_pred),
                "fit_seconds": fit_seconds,
                "iterations": getattr(estimator, "n_iter_", np.nan),
                "n_splits": N_SPLITS,
                "n_repeats": N_REPEATS,
                "random_state": SEED,
            }
            pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
            print(
                f"fit {number}/{len(pending)} fold {fold}/{FOLDS} | "
                f"{name} degree={degree} alpha={alpha:.8g} "
                f"l1={ratio} valid_mse={row['validation_mse']:.6f}",
                flush=True,
            )
        print(
            f"COMPLETE {number}/{len(pending)} | {name} degree={degree} "
            f"elapsed={perf_counter() - start_candidate:.1f}s",
            flush=True,
        )

    data = pd.read_csv(path)
    summary = (
        data.groupby(["candidate_key", "model", "degree", "alpha", "l1_ratio"], dropna=False)
        .agg(
            cv_mse=("validation_mse", "mean"),
            cv_mse_sd=("validation_mse", "std"),
            cv_r2=("validation_r2", "mean"),
            cv_r2_sd=("validation_r2", "std"),
            train_mse=("train_mse", "mean"),
            train_r2=("train_r2", "mean"),
            mean_fit_seconds=("fit_seconds", "mean"),
            mean_iterations=("iterations", "mean"),
            folds=("fold", "nunique"),
        )
        .reset_index()
    )
    summary = summary[summary.folds == FOLDS].sort_values("cv_mse")
    summary.to_csv(ROOT / "var2_regularizer_summary.csv", index=False)
    summary.groupby("model", as_index=False).first().to_csv(
        ROOT / "var2_best_by_regularizer.csv", index=False
    )
    print("\nBest complete configuration by regularizer:", flush=True)
    print(summary.groupby("model", as_index=False).first().to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
