"""Finalize var2 regularizer comparison through degree 17."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
FOLDS = 15
MAX_REPORTED_DEGREE = 17


def main():
    data = pd.read_csv(ROOT / "var2_regularizer_fold_results.csv")
    data = data[data.degree <= MAX_REPORTED_DEGREE]
    keys = ["candidate_key", "model", "degree", "alpha", "l1_ratio"]
    complete = data.groupby("candidate_key")["fold"].nunique()
    complete = complete[complete == FOLDS].index
    data = data[data.candidate_key.isin(complete)]
    summary = (
        data.groupby(keys, dropna=False)
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
    summary.to_csv(ROOT / "var2_regularizer_summary.csv", index=False)
    summary.groupby("model", as_index=False).first().to_csv(
        ROOT / "var2_best_by_regularizer.csv", index=False
    )
    print("Best by regularizer through degree 17:")
    print(summary.groupby("model", as_index=False).first().to_string(index=False))
    print("\nOverall winner:")
    print(summary.head(1).to_string(index=False))


if __name__ == "__main__":
    main()
