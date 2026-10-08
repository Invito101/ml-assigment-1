"""Report the strongest completed var2 configurations from saved artifacts."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
FOLDS = 15


def main():
    fold_path = ROOT / "var2_fold_results.csv"
    if fold_path.exists():
        data = pd.read_csv(fold_path)
        keys = ["stage", "model", "degree", "alpha", "l1_ratio"]
        complete = data.groupby(keys, dropna=False).fold.nunique().reset_index(name="folds")
        complete = complete[complete.folds == FOLDS].drop(columns="folds")
        data = data.merge(complete, on=keys, how="inner")
        summary = (
            data.groupby(keys, dropna=False)
            .agg(cv_mse=("validation_mse", "mean"), cv_r2=("validation_r2", "mean"), folds=("fold", "nunique"))
            .reset_index()
            .sort_values("cv_mse")
        )
    else:
        summary = pd.read_csv(ROOT / "var2_model_selection.csv")
    summary.to_csv(ROOT / "var2_completed_model_comparison.csv", index=False)
    print(summary.head(10).to_string(index=False))
    print("\nBest completed model:")
    print(summary.head(1).to_string(index=False))


if __name__ == "__main__":
    main()
