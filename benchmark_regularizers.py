"""Compare ridge, lasso, and elastic net polynomial regressors."""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")

SEARCH = {
    "var1": (range(4, 8), np.logspace(-3, 2, 13)),
    "var2": (range(10, 15), np.logspace(-4, 1, 13)),
}


def candidates(degree, alpha):
    polynomial = PolynomialFeatures(degree, include_bias=False)
    return {
        "ridge": make_pipeline(polynomial, Ridge(alpha=alpha)),
        "lasso": make_pipeline(
            PolynomialFeatures(degree, include_bias=False),
            StandardScaler(),
            Lasso(alpha=alpha, max_iter=100000),
        ),
        "elastic_net": make_pipeline(
            PolynomialFeatures(degree, include_bias=False),
            StandardScaler(),
            ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=100000),
        ),
    }


def benchmark(problem):
    data = pd.read_csv(f"IMT2024028_train_{problem}.csv")
    X, y = data.drop(columns="y"), data["y"]
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=2026)
    results = []
    degrees, alphas = SEARCH[problem]
    for degree in degrees:
        for alpha in alphas:
            for name, model in candidates(degree, alpha).items():
                scores = cross_validate(
                    model,
                    X,
                    y,
                    cv=cv,
                    scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
                    n_jobs=-1,
                )
                mse = -scores["test_mse"]
                results.append(
                    {
                        "model": name,
                        "degree": degree,
                        "alpha": alpha,
                        "mse": mse.mean(),
                        "r2": scores["test_r2"].mean(),
                        "mse_sd": mse.std(),
                    }
                )
    results = pd.DataFrame(results).sort_values("mse")
    results.to_csv(f"{problem}_regularizer_comparison.csv", index=False)
    print(f"\n{problem}")
    for name in ("ridge", "lasso", "elastic_net"):
        best = results[results.model == name].iloc[0]
        print(
            f"{name}: degree={int(best.degree)}, alpha={best.alpha:.8g}, "
            f"MSE={best.mse:.8f}, R2={best.r2:.8f}, SD={best.mse_sd:.8f}"
        )


if __name__ == "__main__":
    for problem in SEARCH:
        benchmark(problem)