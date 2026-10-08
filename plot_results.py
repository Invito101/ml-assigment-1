"""Create human-readable diagnostic plots for the fitted assignment models."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent
ROLL_NUMBER = "IMT2024028"
CONFIG = {
    "var1": {"degree": 5, "alpha": 0.007943282347242814, "model": "lasso"},
    "var2": {"degree": 12, "alpha": 0.1778279410038923, "model": "ridge"},
}


def make_model(problem):
    settings = CONFIG[problem]
    steps = [PolynomialFeatures(settings["degree"], include_bias=False)]
    if settings["model"] == "lasso":
        steps.extend([StandardScaler(), Lasso(alpha=settings["alpha"], max_iter=100000)])
    else:
        steps.append(Ridge(alpha=settings["alpha"]))
    return make_pipeline(*steps)


def plot_problem(problem):
    train = pd.read_csv(ROOT / f"{ROLL_NUMBER}_train_{problem}.csv")
    test = pd.read_csv(ROOT / f"{ROLL_NUMBER}_test_{problem}.csv")
    pred_path = ROOT / f"{ROLL_NUMBER}_pred_{problem}.csv"
    if not pred_path.exists():
        pred_path = ROOT / f"{ROLL_NUMBER} pred {problem}.csv"
    prediction = pd.read_csv(pred_path)["y"]
    features = [column for column in train.columns if column != "y"]
    X, y = train[features], train["y"]

    # Out-of-fold predictions provide a less optimistic visual check than fitting
    # and plotting predictions on the same rows used for training.
    cv = KFold(n_splits=5, shuffle=True, random_state=2026)
    oof = cross_val_predict(make_model(problem), X, y, cv=cv, n_jobs=-1)
    residuals = y.to_numpy() - oof
    low = min(y.min(), oof.min())
    high = max(y.max(), oof.max())
    padding = 0.05 * (high - low)

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    figure.suptitle(
        f"{problem}: observed data, validation fit, and test predictions",
        fontsize=16,
        fontweight="bold",
    )

    axes[0, 0].hist(y, bins=30, color="#176b87", alpha=0.85, edgecolor="white")
    axes[0, 0].set_title("Training target distribution")
    axes[0, 0].set_xlabel("Observed y")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(alpha=0.2)

    axes[0, 1].scatter(y, oof, s=18, alpha=0.65, color="#d95f02", edgecolors="none")
    axes[0, 1].plot([low - padding, high + padding], [low - padding, high + padding],
                    "k--", linewidth=1.2, label="perfect prediction")
    axes[0, 1].set_title(
        f"5-fold out-of-fold predictions\nMSE={mean_squared_error(y, oof):.3f}, "
        f"R2={r2_score(y, oof):.3f}"
    )
    axes[0, 1].set_xlabel("Observed y")
    axes[0, 1].set_ylabel("Predicted y")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(alpha=0.2)

    axes[1, 0].scatter(oof, residuals, s=18, alpha=0.65, color="#5e3c99", edgecolors="none")
    axes[1, 0].axhline(0, color="black", linestyle="--", linewidth=1.2)
    axes[1, 0].set_title("Out-of-fold residuals")
    axes[1, 0].set_xlabel("Predicted y")
    axes[1, 0].set_ylabel("Observed - predicted")
    axes[1, 0].grid(alpha=0.2)

    axes[1, 1].plot(np.arange(len(prediction)), prediction, color="#238b45", linewidth=1)
    axes[1, 1].set_title("Predictions on the hidden-target test rows")
    axes[1, 1].set_xlabel("Test row index")
    axes[1, 1].set_ylabel("Predicted y")
    axes[1, 1].grid(alpha=0.2)

    figure.text(
        0.5,
        0.01,
        f"Inputs: {', '.join(features)} | Test predictions: {len(test)} rows",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.95))
    output = ROOT / f"{problem}_diagnostics.png"
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"saved {output.name} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    for problem in CONFIG:
        plot_problem(problem)