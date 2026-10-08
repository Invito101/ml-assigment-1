"""Plot completed var1 degree-curve evidence for the report."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
FOLDS = 15


def load_complete_degree_curves():
    data = pd.read_csv(ROOT / "var1_fold_results.csv")
    data = data[data["stage"] == "degree_curve"].copy()
    keys = ["stage", "model", "degree", "alpha", "l1_ratio"]
    complete = (
        data.groupby(keys, dropna=False)["fold"]
        .nunique()
        .loc[lambda counts: counts == FOLDS]
        .reset_index()[keys]
    )
    data = data.merge(complete, on=keys, how="inner")
    summary = (
        data.groupby(keys, dropna=False)
        .agg(
            validation_mse=("validation_mse", "mean"),
            validation_mse_sd=("validation_mse", "std"),
            validation_r2=("validation_r2", "mean"),
            validation_r2_sd=("validation_r2", "std"),
            train_mse=("train_mse", "mean"),
            train_r2=("train_r2", "mean"),
            mean_fit_seconds=("fit_seconds", "mean"),
            mean_iterations=("iterations", "mean"),
            folds=("fold", "nunique"),
        )
        .reset_index()
        .sort_values(["model", "degree", "l1_ratio"], na_position="first")
    )
    summary.to_csv(ROOT / "var1_degree_curve_summary.csv", index=False)
    return summary


def label(row):
    if row["model"] == "elastic_net":
        return f"Elastic Net (l1={row['l1_ratio']:.1f})"
    return row["model"].title()


def plot_metric(summary, value, ylabel, filename, title):
    figure, axis = plt.subplots(figsize=(11, 7))
    for _, group in summary.groupby(["model", "l1_ratio"], dropna=False):
        group = group.sort_values("degree")
        axis.plot(group["degree"], group[value], marker="o", linewidth=2, label=label(group.iloc[0]))
    axis.set_title(title, fontsize=15, fontweight="bold")
    axis.set_xlabel("Polynomial degree")
    axis.set_ylabel(ylabel)
    axis.set_xticks(range(1, 11))
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(ROOT / filename, dpi=180)
    plt.close(figure)


def main():
    summary = load_complete_degree_curves()
    plot_metric(
        summary,
        "validation_mse",
        "Mean validation MSE",
        "var1_degree_validation_mse.png",
        "var1 degree comparison: validation MSE",
    )
    plot_metric(
        summary,
        "validation_r2",
        "Mean validation R2",
        "var1_degree_validation_r2.png",
        "var1 degree comparison: validation R2",
    )

    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    for _, group in summary.groupby(["model", "l1_ratio"], dropna=False):
        group = group.sort_values("degree")
        name = label(group.iloc[0])
        axes[0].plot(group["degree"], group["train_mse"], marker="o", label=f"{name} train")
        axes[0].plot(group["degree"], group["validation_mse"], linestyle="--", label=f"{name} validation")
        axes[1].plot(group["degree"], group["mean_fit_seconds"], marker="o", label=name)
    axes[0].set_title("Training vs validation MSE", fontweight="bold")
    axes[0].set_xlabel("Polynomial degree")
    axes[0].set_ylabel("MSE")
    axes[1].set_title("Mean fit time per fold", fontweight="bold")
    axes[1].set_xlabel("Polynomial degree")
    axes[1].set_ylabel("Seconds")
    for axis in axes:
        axis.set_xticks(range(1, 11))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(ROOT / "var1_degree_training_and_runtime.png", dpi=180)
    plt.close(figure)

    print(f"Saved {len(summary)} complete degree-curve summaries.")
    print(summary.sort_values("validation_mse").head(10).to_string(index=False))


if __name__ == "__main__":
    main()
