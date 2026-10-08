"""Resumable var2 Ridge-only degree and alpha experiment.

Stage 1 screens Ridge degrees 1--20. Stage 2 tunes Ridge alpha only at the
inferred best degree. Fold results are appended immediately.
"""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

ROOT = Path(__file__).resolve().parent
SEED = 2026
N_SPLITS = 5
N_REPEATS = 3
FOLDS = N_SPLITS * N_REPEATS
MAX_DEGREE = 17
ALPHAS = np.logspace(-3, 1, 33)
TUNING_DEGREES = range(8, 18)


def make_model(degree, alpha):
    return make_pipeline(
        PolynomialFeatures(degree, include_bias=False),
        Ridge(alpha=alpha),
    )


def key(stage, degree, alpha):
    return stage, int(degree), round(float(alpha), 8)


def completed(path):
    if not path.exists():
        return set()
    data = pd.read_csv(path)
    counts = data.groupby(["stage", "degree", "alpha"])["fold"].nunique()
    return {key(stage, degree, alpha) for (stage, degree, alpha), count in counts.items() if count == FOLDS}


def run(candidates, X, y, splitter, path, done):
    pending = [candidate for candidate in candidates if key(*candidate) not in done]
    print(f"{candidates[0][0]}: {len(candidates)} candidates, {len(pending)} pending, {len(pending) * FOLDS} fits.", flush=True)
    for number, (stage, degree, alpha) in enumerate(pending, start=1):
        start_candidate = perf_counter()
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(X), start=1):
            model = make_model(degree, alpha)
            start_fit = perf_counter()
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            fit_seconds = perf_counter() - start_fit
            train_pred = model.predict(X.iloc[train_idx])
            valid_pred = model.predict(X.iloc[valid_idx])
            row = {
                "stage": stage, "model": "ridge", "degree": degree, "alpha": alpha,
                "l1_ratio": np.nan, "fold": fold,
                "train_mse": mean_squared_error(y.iloc[train_idx], train_pred),
                "train_r2": r2_score(y.iloc[train_idx], train_pred),
                "validation_mse": mean_squared_error(y.iloc[valid_idx], valid_pred),
                "validation_r2": r2_score(y.iloc[valid_idx], valid_pred),
                "fit_seconds": fit_seconds, "iterations": np.nan,
            }
            pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
            print(
                f"fit candidate {number}/{len(pending)} fold {fold}/{FOLDS} | "
                f"stage={stage} degree={degree} alpha={alpha:.8g} "
                f"valid_mse={row['validation_mse']:.6f}", flush=True,
            )
        print(f"COMPLETE {stage} degree={degree} alpha={alpha:.8g} elapsed={perf_counter() - start_candidate:.1f}s", flush=True)
    return completed(path)


def summary(path):
    data = pd.read_csv(path)
    return (
        data.groupby(["stage", "model", "degree", "alpha"], dropna=False)
        .agg(cv_mse=("validation_mse", "mean"), cv_r2=("validation_r2", "mean"), folds=("fold", "nunique"))
        .reset_index()
        .query("folds == @FOLDS")
        .sort_values("cv_mse")
    )


def main():
    train = pd.read_csv(ROOT / "IMT2024028_train_var2.csv")
    X, y = train.drop(columns="y"), train["y"]
    path = ROOT / "var2_ridge_fold_results.csv"
    splitter = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    done = completed(path)
    degree_candidates = [("degree_screen", degree, 1e-8) for degree in range(1, MAX_DEGREE + 1)]
    done = run(degree_candidates, X, y, splitter, path, done)
    degree_summary = summary(path)
    screen_summary = degree_summary[degree_summary.stage == "degree_screen"]
    best_degree = int(screen_summary.loc[screen_summary.cv_mse.idxmin(), "degree"])
    print(f"INFERRED BEST DEGREE (SCREEN): {best_degree}", flush=True)
    # Ridge regularization can change which degree generalizes best, so tune
    # the high-degree region around the screen winner rather than only fixing
    # alpha at the single unregularized winner.
    alpha_candidates = [
        ("alpha_tuning", degree, alpha)
        for degree in TUNING_DEGREES
        for alpha in ALPHAS
    ]
    run(alpha_candidates, X, y, splitter, path, done)
    result = summary(path)
    result.to_csv(ROOT / "var2_ridge_summary.csv", index=False)
    print("\nBest Ridge configuration:")
    print(result.head(1).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
