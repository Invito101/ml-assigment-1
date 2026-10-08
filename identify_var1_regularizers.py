"""Identify the strongest completed var1 regularizer from saved fold results."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
FOLDS = 15


def main():
    data = pd.read_csv(ROOT / "var1_fold_results.csv")
    keys = ["stage", "model", "degree", "alpha", "l1_ratio"]
    complete = data.groupby(keys, dropna=False)["fold"].nunique().reset_index(name="folds")
    complete = complete[complete["folds"] == FOLDS].drop(columns="folds")
    data = data.merge(
        complete,
        on=keys,
        how="inner",
    )
    summary = (
        data.groupby(["stage", "model", "degree", "alpha", "l1_ratio"], dropna=False)
        .agg(
            cv_mse=("validation_mse", "mean"),
            cv_r2=("validation_r2", "mean"),
            cv_mse_sd=("validation_mse", "std"),
            mean_iterations=("iterations", "mean"),
            folds=("fold", "nunique"),
        )
        .reset_index()
        .sort_values("cv_mse")
    )
    summary.to_csv(ROOT / "var1_completed_model_comparison.csv", index=False)
    degree_five = summary[summary.degree == 5]
    best_by_model = degree_five.loc[degree_five.groupby("model").cv_mse.idxmin()]
    best_by_model.to_csv(ROOT / "var1_completed_degree5_comparison.csv", index=False)
    print("Best completed degree-5 configuration by model:")
    print(best_by_model.to_string(index=False))
    print("\nBest completed configuration overall:")
    print(summary.head(1).to_string(index=False))


if __name__ == "__main__":
    main()
