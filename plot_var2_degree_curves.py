"""Plot completed var2 degree-curve evidence for the report.

Produces exact counterparts to var1 degree curve figures:
1. var2_degree_validation_mse.png
2. var2_degree_validation_r2.png
3. var2_degree_training_and_runtime.png
4. var2_degree_curve_summary.csv
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FOLDS = 15
MAX_DEGREE = 17


def load_complete_degree_curves():
    # Load regularizer baseline comparison (degrees 1--17)
    reg_data = pd.read_csv(ROOT / "var2_regularizer_fold_results.csv")
    reg_data = reg_data[reg_data.degree <= MAX_DEGREE].copy()
    
    keys = ["model", "degree", "alpha", "l1_ratio"]
    complete = (
        reg_data.groupby(keys, dropna=False)["fold"]
        .nunique()
        .loc[lambda c: c == FOLDS]
        .reset_index()[keys]
    )
    reg_data = reg_data.merge(complete, on=keys, how="inner")
    
    summary = (
        reg_data.groupby(keys, dropna=False)
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
    )
    
    # Also add the tuned Ridge curve from var2_ridge_summary.csv (degrees 8--17)
    ridge_summary = pd.read_csv(ROOT / "var2_ridge_summary.csv")
    tuning = ridge_summary[ridge_summary.stage == "alpha_tuning"]
    best_tuned = tuning.loc[tuning.groupby("degree")["cv_mse"].idxmin()].copy()
    
    tuned_rows = []
    # Fetch train_mse and fit_seconds for the tuned ridge from var2_ridge_fold_results.csv
    ridge_folds = pd.read_csv(ROOT / "var2_ridge_fold_results.csv")
    for _, r in best_tuned.iterrows():
        deg, a = int(r["degree"]), float(r["alpha"])
        sub = ridge_folds[(ridge_folds.stage == "alpha_tuning") & 
                          (ridge_folds.degree == deg) & 
                          (np.isclose(ridge_folds.alpha, a))]
        if len(sub) == FOLDS:
            tuned_rows.append({
                "model": "tuned_ridge",
                "degree": deg,
                "alpha": a,
                "l1_ratio": np.nan,
                "validation_mse": sub["validation_mse"].mean(),
                "validation_mse_sd": sub["validation_mse"].std(),
                "validation_r2": sub["validation_r2"].mean(),
                "validation_r2_sd": sub["validation_r2"].std(),
                "train_mse": sub["train_mse"].mean(),
                "train_r2": sub["train_r2"].mean(),
                "mean_fit_seconds": sub["fit_seconds"].mean(),
                "mean_iterations": np.nan,
                "folds": FOLDS,
            })
    
    if tuned_rows:
        summary = pd.concat([summary, pd.DataFrame(tuned_rows)], ignore_index=True)
    
    summary = summary.sort_values(["model", "degree", "l1_ratio"], na_position="first")
    summary.to_csv(ROOT / "var2_degree_curve_summary.csv", index=False)
    return summary


def label(row):
    if row["model"] == "elastic_net":
        return f"Elastic Net (l1={row['l1_ratio']:.1f})"
    if row["model"] == "tuned_ridge":
        return "Tuned Ridge (Optimal alpha)"
    if row["model"] == "ridge":
        return "Unregularized Ridge (alpha=1e-8)"
    return row["model"].title()


def plot_mse(summary):
    figure, axis = plt.subplots(figsize=(11, 7))
    for _, group in summary.groupby(["model", "l1_ratio"], dropna=False):
        group = group.sort_values("degree")
        name = label(group.iloc[0])
        lw = 2.5 if "Tuned" in name else 1.8
        axis.plot(group["degree"], group["validation_mse"], marker="o", linewidth=lw, label=name)
    
    axis.axvline(12, color="green", linestyle="--", alpha=0.7, label="Selected Model (d=12)")
    axis.set_title("var2 degree comparison: validation MSE (Degrees 1–17)", fontsize=15, fontweight="bold")
    axis.set_xlabel("Polynomial degree", fontsize=12)
    axis.set_ylabel("Mean validation MSE", fontsize=12)
    axis.set_xticks(range(1, 18))
    axis.set_ylim(0, 4.0)  # Focus on competitive region while showing divergence of unregularized
    axis.grid(alpha=0.25)
    axis.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    figure.tight_layout()
    figure.savefig(ROOT / "var2_degree_validation_mse.png", dpi=180)
    plt.close(figure)
    print("Saved var2_degree_validation_mse.png")


def plot_r2(summary):
    figure, axis = plt.subplots(figsize=(11, 7))
    for _, group in summary.groupby(["model", "l1_ratio"], dropna=False):
        group = group.sort_values("degree")
        name = label(group.iloc[0])
        lw = 2.5 if "Tuned" in name else 1.8
        axis.plot(group["degree"], group["validation_r2"], marker="o", linewidth=lw, label=name)
        
    axis.axvline(12, color="green", linestyle="--", alpha=0.7, label="Selected Model (d=12)")
    axis.set_title("var2 degree comparison: validation R² (Degrees 1–17)", fontsize=15, fontweight="bold")
    axis.set_xlabel("Polynomial degree", fontsize=12)
    axis.set_ylabel("Mean validation R²", fontsize=12)
    axis.set_xticks(range(1, 18))
    axis.set_ylim(0.2, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    figure.tight_layout()
    figure.savefig(ROOT / "var2_degree_validation_r2.png", dpi=180)
    plt.close(figure)
    print("Saved var2_degree_validation_r2.png")


def plot_training_and_runtime(summary):
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    for _, group in summary.groupby(["model", "l1_ratio"], dropna=False):
        group = group.sort_values("degree")
        name = label(group.iloc[0])
        axes[0].plot(group["degree"], group["train_mse"], marker="o", label=f"{name} train")
        axes[0].plot(group["degree"], group["validation_mse"], linestyle="--", label=f"{name} val")
        axes[1].plot(group["degree"], group["mean_fit_seconds"], marker="o", label=name)
        
    axes[0].set_title("Training vs Validation MSE (Degrees 1–17)", fontweight="bold")
    axes[0].set_xlabel("Polynomial degree")
    axes[0].set_ylabel("MSE")
    axes[0].set_ylim(0, 3.0)
    
    axes[1].set_title("Mean fit time per fold (Seconds)", fontweight="bold")
    axes[1].set_xlabel("Polynomial degree")
    axes[1].set_ylabel("Seconds")
    
    for axis in axes:
        axis.set_xticks(range(1, 18))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7.5, frameon=True, facecolor="white", edgecolor="#cccccc")
        
    figure.tight_layout()
    figure.savefig(ROOT / "var2_degree_training_and_runtime.png", dpi=180)
    plt.close(figure)
    print("Saved var2_degree_training_and_runtime.png")


def main():
    summary = load_complete_degree_curves()
    plot_mse(summary)
    plot_r2(summary)
    plot_training_and_runtime(summary)
    print("All var2 figures generated successfully!")


if __name__ == "__main__":
    main()
