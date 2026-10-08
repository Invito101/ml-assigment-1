"""Reproducible polynomial-model selection, fitting, and inference.

The search is explicit: every degree, regularizer, penalty value, fold setup,
and validation result is saved before the final refit.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parent
ROLL_NUMBER = "IMT2024028"
N_SPLITS = 5
N_REPEATS = 3
RANDOM_STATE = 2026

SEARCH = {
    "var1": {
        "max_degree": 10,
        "refinement_degrees": range(3, 9),
        "ridge_alphas": np.logspace(-1, 1, 9),
        # Dense local search around the best lasso region found in the initial
        # comparison, plus a wider range to guard against a local choice.
        "sparse_alphas": np.unique(
            np.r_[np.logspace(-4, -2, 21), np.linspace(0.011, 0.03, 20)]
        ),
    },
    "var2": {
        "max_degree": 20,
        "refinement_degrees": range(10, 15),
        "ridge_alphas": np.logspace(-1.5, 0.5, 9),
        "sparse_alphas": np.logspace(-2, -0.5, 7),
    },
}
L1_RATIOS = (0.1, 0.5, 0.9)


def make_model(model_name, degree, alpha, l1_ratio=None):
    polynomial = PolynomialFeatures(degree=degree, include_bias=False)
    if model_name == "ridge":
        return make_pipeline(polynomial, Ridge(alpha=alpha))
    if model_name == "lasso":
        return make_pipeline(polynomial, StandardScaler(), Lasso(alpha=alpha, max_iter=100000))
    if model_name == "elastic_net":
        return make_pipeline(
            polynomial,
            StandardScaler(),
            ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=100000),
        )
    raise ValueError(f"Unknown model: {model_name}")


def evaluate(model, X, y, cv):
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
        n_jobs=-1,
    )
    fold_mse = -scores["test_mse"]
    return fold_mse.mean(), fold_mse.std(), scores["test_r2"].mean()


def candidate_rows(problem, X, y, cv):
    settings = SEARCH[problem]
    rows = []
    for degree in range(1, settings["max_degree"] + 1):
        mse, mse_sd, r2 = evaluate(make_model("ridge", degree, 1e-8), X, y, cv)
        rows.append({"stage": "degree_screen", "model": "ridge", "degree": degree,
                     "alpha": 1e-8, "l1_ratio": np.nan, "cv_mse": mse,
                     "cv_mse_sd": mse_sd, "cv_r2": r2})

    for degree in settings["refinement_degrees"]:
        for alpha in settings["ridge_alphas"]:
            mse, mse_sd, r2 = evaluate(make_model("ridge", degree, alpha), X, y, cv)
            rows.append({"stage": "regularizer_search", "model": "ridge", "degree": degree,
                         "alpha": alpha, "l1_ratio": np.nan, "cv_mse": mse,
                         "cv_mse_sd": mse_sd, "cv_r2": r2})
        for alpha in settings["sparse_alphas"]:
            mse, mse_sd, r2 = evaluate(make_model("lasso", degree, alpha), X, y, cv)
            rows.append({"stage": "regularizer_search", "model": "lasso", "degree": degree,
                         "alpha": alpha, "l1_ratio": np.nan, "cv_mse": mse,
                         "cv_mse_sd": mse_sd, "cv_r2": r2})
            for l1_ratio in L1_RATIOS:
                mse, mse_sd, r2 = evaluate(
                    make_model("elastic_net", degree, alpha, l1_ratio), X, y, cv
                )
                rows.append({"stage": "regularizer_search", "model": "elastic_net", "degree": degree,
                             "alpha": alpha, "l1_ratio": l1_ratio, "cv_mse": mse,
                             "cv_mse_sd": mse_sd, "cv_r2": r2})
    return pd.DataFrame(rows)


def run_problem(problem, search=False):
    train = pd.read_csv(ROOT / f"{ROLL_NUMBER}_train_{problem}.csv")
    test = pd.read_csv(ROOT / f"{ROLL_NUMBER}_test_{problem}.csv")
    if train.isna().any().any() or test.isna().any().any():
        raise ValueError(f"Missing values found in {problem}")
    if train.duplicated().any():
        raise ValueError(f"Duplicate rows found in {problem} training data")
    feature_columns = [column for column in train.columns if column != "y"]
    if list(test.columns) != feature_columns:
        raise ValueError(f"Unexpected test columns for {problem}")

    X, y = train[feature_columns], train["y"]
    cv = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    if search:
        results = candidate_rows(problem, X, y, cv)
        results.insert(0, "problem", problem)
        results["n_splits"] = N_SPLITS
        results["n_repeats"] = N_REPEATS
        results["random_state"] = RANDOM_STATE
        results.to_csv(ROOT / f"{problem}_model_selection.csv", index=False)
        candidates = results[results["stage"] == "regularizer_search"]
        selected = candidates.loc[candidates["cv_mse"].idxmin()]
    else:
        saved = pd.read_csv(ROOT / "selected_models.csv")
        selected = saved.loc[saved["problem"] == problem].iloc[0]
    model_name = selected["model"]
    degree = int(selected["degree"])
    alpha = float(selected["alpha"])
    l1_ratio = None if pd.isna(selected["l1_ratio"]) else float(selected["l1_ratio"])
    model = make_model(model_name, degree, alpha, l1_ratio)
    model.fit(X, y)
    predictions = model.predict(test[feature_columns])
    output = pd.DataFrame({"y": predictions})
    if len(output) != len(test) or not np.isfinite(output["y"]).all():
        raise ValueError(f"Invalid prediction output for {problem}")
    output.to_csv(ROOT / f"{ROLL_NUMBER}_pred_{problem}.csv", index=False)

    cv_mse, cv_mse_sd, cv_r2 = evaluate(model, X, y, cv)
    return {"problem": problem, "model": model_name, "degree": degree, "alpha": alpha,
            "l1_ratio": l1_ratio, "cv_mse": float(cv_mse),
            "cv_mse_sd": float(cv_mse_sd), "cv_r2": float(cv_r2),
            "train_mse": mean_squared_error(y, model.predict(X)),
            "train_r2": r2_score(y, model.predict(X)), "prediction_rows": len(predictions)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--search",
        action="store_true",
        help="run the full repeated-CV degree and regularizer search",
    )
    args = parser.parse_args()
    summaries = [run_problem(problem, search=args.search) for problem in ("var1", "var2")]
    pd.DataFrame(summaries).to_csv(ROOT / "final_model_summary.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
