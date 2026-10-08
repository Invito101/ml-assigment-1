"""Focused, resumable var2 experiment for report-ready evidence.

Stage 1 evaluates every permitted degree with baseline Ridge, Lasso, and
Elastic Net models. Stage 2 tunes alpha only for the strongest degrees found
in Stage 1. Every fold is appended immediately to var2_fold_results.csv.
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
MAX_DEGREE = 20
FOLDS_PER_CANDIDATE = N_SPLITS * N_REPEATS

BASELINE = {
    "ridge": {"alpha": 1e-8, "l1_ratio": np.nan},
    "lasso": {"alpha": 0.01, "l1_ratio": np.nan},
    "elastic_net": {"alpha": 0.01, "l1_ratio": 0.5},
}
RIDGE_ALPHAS = np.logspace(-2, 1, 9)
SPARSE_ALPHAS = np.logspace(-2, -0.5, 7)
L1_RATIOS = (0.1, 0.5, 0.9)


def make_model(model_name, degree, alpha, l1_ratio=np.nan):
    polynomial = PolynomialFeatures(degree=degree, include_bias=False)
    if model_name == "ridge":
        return make_pipeline(polynomial, Ridge(alpha=alpha))
    if model_name == "lasso":
        return make_pipeline(
            polynomial,
            StandardScaler(),
            Lasso(alpha=alpha, max_iter=100000),
        )
    return make_pipeline(
        polynomial,
        StandardScaler(),
        ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=100000),
    )


def candidate_id(stage, model, degree, alpha, l1_ratio):
    ratio = "none" if pd.isna(l1_ratio) else f"{l1_ratio:.8g}"
    return f"{stage}|{model}|d{degree}|a{alpha:.12g}|l{ratio}"


def make_candidate(stage, model, degree, alpha, l1_ratio=np.nan):
    return {
        "candidate_id": candidate_id(stage, model, degree, alpha, l1_ratio),
        "stage": stage,
        "model": model,
        "degree": int(degree),
        "alpha": float(alpha),
        "l1_ratio": l1_ratio,
    }


def degree_candidates():
    for degree in range(1, MAX_DEGREE + 1):
        for model, values in BASELINE.items():
            yield make_candidate(
                "degree_screen", model, degree, values["alpha"], values["l1_ratio"]
            )


def completed_ids(path):
    if not path.exists():
        return set()
    data = pd.read_csv(path)
    counts = data.groupby("candidate_id")["fold"].nunique()
    return set(counts[counts == FOLDS_PER_CANDIDATE].index)


def summarize(fold_path):
    data = pd.read_csv(fold_path)
    return (
        data.groupby(
            ["candidate_id", "stage", "model", "degree", "alpha", "l1_ratio"],
            dropna=False,
        )
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
        .sort_values("cv_mse")
    )


def focused_candidates(fold_path):
    summary = summarize(fold_path)
    degree_results = summary[summary["stage"] == "degree_screen"]
    selected_degrees = set()
    for model in ("ridge", "lasso", "elastic_net"):
        best = degree_results[degree_results["model"] == model].nsmallest(2, "cv_mse")
        for degree in best["degree"].astype(int):
            selected_degrees.update({degree - 1, degree, degree + 1})
    selected_degrees = sorted(degree for degree in selected_degrees if 1 <= degree <= MAX_DEGREE)

    for degree in selected_degrees:
        for alpha in RIDGE_ALPHAS:
            yield make_candidate("alpha_tuning", "ridge", degree, alpha)
        for alpha in SPARSE_ALPHAS:
            yield make_candidate("alpha_tuning", "lasso", degree, alpha)
            for ratio in L1_RATIOS:
                yield make_candidate("alpha_tuning", "elastic_net", degree, alpha, ratio)


def run_candidate(candidate, X, y, splitter, fold_path, finished, progress, total):
    start_candidate = perf_counter()
    splits = list(splitter.split(X))
    for fold_id, (train_idx, valid_idx) in enumerate(splits, start=1):
        model = make_model(
            candidate["model"],
            candidate["degree"],
            candidate["alpha"],
            candidate["l1_ratio"],
        )
        start_fit = perf_counter()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        fit_seconds = perf_counter() - start_fit
        train_pred = model.predict(X.iloc[train_idx])
        valid_pred = model.predict(X.iloc[valid_idx])
        estimator = model.steps[-1][1]
        row = {
            **candidate,
            "fold": fold_id,
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
        pd.DataFrame([row]).to_csv(
            fold_path,
            mode="a",
            header=not fold_path.exists(),
            index=False,
        )
        finished += 1
        print(
            f"fit {finished}/{total} | {candidate['stage']} "
            f"{candidate['model']} degree={candidate['degree']} "
            f"alpha={candidate['alpha']:.8g} fold={fold_id}/{FOLDS_PER_CANDIDATE} "
            f"valid_mse={row['validation_mse']:.6f} "
            f"elapsed={perf_counter() - start_candidate:.1f}s",
            flush=True,
        )
    print(
        f"COMPLETE {progress} | {candidate['stage']} {candidate['model']} "
        f"degree={candidate['degree']} alpha={candidate['alpha']:.8g} "
        f"elapsed={perf_counter() - start_candidate:.1f}s",
        flush=True,
    )
    return finished


def main():
    data = pd.read_csv(ROOT / "IMT2024028_train_var2.csv")
    X, y = data.drop(columns="y"), data["y"]
    fold_path = ROOT / "var2_fold_results.csv"
    finished_ids = completed_ids(fold_path)
    splitter = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)

    stage_one = list(degree_candidates())
    if not all(candidate["candidate_id"] in finished_ids for candidate in stage_one):
        candidates = stage_one
        print(
            f"Stage 1: {len(candidates)} candidates, "
            f"{len(candidates) * FOLDS_PER_CANDIDATE} fits.",
            flush=True,
        )
    else:
        candidates = list(focused_candidates(fold_path))
        print(
            f"Stage 1 complete. Stage 2: {len(candidates)} focused candidates, "
            f"{len(candidates) * FOLDS_PER_CANDIDATE} fits.",
            flush=True,
        )

    pending = [candidate for candidate in candidates if candidate["candidate_id"] not in finished_ids]
    total = sum(len(stage_one) if candidates is stage_one else len(candidates) for _ in [0]) * FOLDS_PER_CANDIDATE
    finished = sum(
        FOLDS_PER_CANDIDATE for candidate in candidates if candidate["candidate_id"] in finished_ids
    )
    for number, candidate in enumerate(pending, start=1):
        finished = run_candidate(
            candidate,
            X,
            y,
            splitter,
            fold_path,
            finished,
            f"{number}/{len(pending)}",
            total,
        )

    summary = summarize(fold_path)
    summary.to_csv(ROOT / "var2_experiment_summary.csv", index=False)
    degree_summary = summary[summary["stage"] == "degree_screen"]
    degree_summary.to_csv(ROOT / "var2_degree_summary.csv", index=False)
    tuned_summary = summary[summary["stage"] == "alpha_tuning"]
    tuned_summary.to_csv(ROOT / "var2_alpha_summary.csv", index=False)
    best_by_model = summary.loc[summary.groupby("model")["cv_mse"].idxmin()]
    best_by_model.to_csv(ROOT / "var2_best_by_model.csv", index=False)
    print("\nBest complete candidate by model:", flush=True)
    print(best_by_model.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
