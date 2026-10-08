"""Resumable degree-5 Lasso alpha sweep for var1.

Existing alpha-sweep rows are preserved and skipped. New fold results are
appended to var1_fold_results.csv immediately.
"""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parent
SEED = 2026
N_SPLITS = 5
N_REPEATS = 3
DEGREE = 5
FOLDS = N_SPLITS * N_REPEATS

# Covers the previous grid, the known useful region, and a dense intermediate
# grid so the final choice is based on an explicit Lasso-only experiment.
ALPHAS = np.unique(
    np.r_[
        np.logspace(-4, -2, 21),
        np.linspace(0.011, 0.03, 20),
        np.array([0.007943282347242814]),
    ]
)


def candidate_key(alpha):
    return ("alpha_sweep_lasso", DEGREE, float(alpha))


def completed_candidates(path):
    if not path.exists():
        return set()
    data = pd.read_csv(path)
    data = data[
        (data.stage == "alpha_sweep")
        & (data.model == "lasso")
        & (data.degree == DEGREE)
    ]
    return {
        candidate_key(alpha)
        for alpha, group in data.groupby("alpha")
        if group.fold.nunique() == FOLDS
    }


def main():
    train = pd.read_csv(ROOT / "IMT2024028_train_var1.csv")
    X, y = train.drop(columns="y"), train["y"]
    fold_path = ROOT / "var1_fold_results.csv"
    splitter = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    completed = completed_candidates(fold_path)
    pending = [alpha for alpha in ALPHAS if candidate_key(alpha) not in completed]
    print(
        f"Lasso-only alpha sweep: {len(ALPHAS)} candidates, "
        f"{len(pending)} pending, {len(pending) * FOLDS} pending fits.",
        flush=True,
    )

    for number, alpha in enumerate(pending, start=1):
        start_candidate = perf_counter()
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(X), start=1):
            model = make_pipeline(
                PolynomialFeatures(DEGREE, include_bias=False),
                StandardScaler(),
                Lasso(alpha=float(alpha), max_iter=100000),
            )
            start_fit = perf_counter()
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            fit_seconds = perf_counter() - start_fit
            train_pred = model.predict(X.iloc[train_idx])
            valid_pred = model.predict(X.iloc[valid_idx])
            row = {
                "stage": "alpha_sweep",
                "model": "lasso",
                "degree": DEGREE,
                "alpha": float(alpha),
                "l1_ratio": np.nan,
                "fold": fold,
                "train_mse": mean_squared_error(y.iloc[train_idx], train_pred),
                "train_r2": r2_score(y.iloc[train_idx], train_pred),
                "validation_mse": mean_squared_error(y.iloc[valid_idx], valid_pred),
                "validation_r2": r2_score(y.iloc[valid_idx], valid_pred),
                "fit_seconds": fit_seconds,
                "iterations": model.steps[-1][1].n_iter_,
            }
            pd.DataFrame([row]).to_csv(
                fold_path,
                mode="a",
                header=not fold_path.exists(),
                index=False,
            )
            print(
                f"fit {number}/{len(pending)} candidate, fold {fold}/{FOLDS} | "
                f"alpha={alpha:.10g} valid_mse={row['validation_mse']:.6f} | "
                f"elapsed={perf_counter() - start_candidate:.1f}s",
                flush=True,
            )
        print(
            f"COMPLETE candidate {number}/{len(pending)} alpha={alpha:.10g} "
            f"elapsed={perf_counter() - start_candidate:.1f}s",
            flush=True,
        )

    data = pd.read_csv(fold_path)
    data = data[(data.stage == "alpha_sweep") & (data.model == "lasso") & (data.degree == DEGREE)]
    summary = (
        data.groupby(["stage", "model", "degree", "alpha"], dropna=False)
        .agg(cv_mse=("validation_mse", "mean"), cv_r2=("validation_r2", "mean"), folds=("fold", "nunique"))
        .reset_index()
    )
    summary = summary[summary.folds == FOLDS].sort_values("cv_mse")
    summary.to_csv(ROOT / "var1_lasso_alpha_summary.csv", index=False)
    print("\nBest completed Lasso alpha:")
    print(summary.head(1).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
