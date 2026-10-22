"""Sanity check added after the phase 4 protocol: do other standard model families beat the
candidates? Same time split, same 6 sensors, fixed settings; plus a small LightGBM tuning grid
(tuned on the calibration window, inside the training period) and an average of three boosters.

Usage (from the repository root, after notebook 02 has produced the clean table):
    python3 scripts/benchmark_other_models.py
Output: reports/model_benchmark_extra.csv. Needs xgboost and catboost (requirements.txt).
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fragrance import config, model


def main():
    df = pd.read_parquet(config.CLEAN_PARQUET)
    data = df[~df["all_sensors_missing"]].reset_index(drop=True)
    y, F = data[config.TARGET], model.FEATURES_NO_DENSITY
    ts = model.time_split(data)
    tr, te = ts["train"], ts["test"]
    impute = lambda: SimpleImputer(strategy="median", add_indicator=True)
    seed = model.SEED
    candidates = {
        "lightgbm (selected candidate)": lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31,
                                                           min_child_samples=100, random_state=seed, verbose=-1),
        "extra trees": make_pipeline(impute(), ExtraTreesRegressor(n_estimators=200, min_samples_leaf=50, max_features=0.6,
                                                                   n_jobs=-1, random_state=seed)),
        "sklearn hist gradient boosting": HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
                                                                        min_samples_leaf=100, random_state=seed),
        "xgboost": xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, min_child_weight=100,
                                    random_state=seed, n_jobs=-1),
        "catboost": cb.CatBoostRegressor(iterations=600, learning_rate=0.05, depth=6, random_seed=seed, verbose=0),
        "neural network (MLP 64-32)": make_pipeline(impute(), StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=200, random_state=seed)),
    }
    rows, preds = {}, {}
    for name, est in candidates.items():
        t0 = time.time()
        est.fit(data.loc[tr, F], y[tr])
        preds[name] = est.predict(data.loc[te, F])
        rows[name] = {**model.regression_metrics(y[te], preds[name]).to_dict(), "fit_seconds": time.time() - t0}
        print(name, round(rows[name]["MAE"], 3), flush=True)

    # Small tuning grid for LightGBM, scored on the calibration window (never on the test set)
    best = None
    for leaves in [15, 31, 63]:
        for min_child in [50, 200]:
            for lr, n in [(0.05, 600), (0.02, 1500)]:
                m = lgb.LGBMRegressor(n_estimators=n, learning_rate=lr, num_leaves=leaves, min_child_samples=min_child,
                                      random_state=seed, verbose=-1).fit(data.loc[ts["fit"], F], y[ts["fit"]])
                mae = np.abs(m.predict(data.loc[ts["calib"], F]) - y[ts["calib"]]).mean()
                if best is None or mae < best[0]:
                    best = (mae, leaves, min_child, lr, n)
    _, leaves, min_child, lr, n = best
    tuned = lgb.LGBMRegressor(n_estimators=n, learning_rate=lr, num_leaves=leaves, min_child_samples=min_child,
                              random_state=seed, verbose=-1).fit(data.loc[tr, F], y[tr])
    rows[f"lightgbm tuned (leaves={leaves}, min_child={min_child}, lr={lr})"] = model.regression_metrics(
        y[te], tuned.predict(data.loc[te, F])).to_dict()
    avg = np.mean([preds[k] for k in ["lightgbm (selected candidate)", "xgboost", "catboost"]], axis=0)
    rows["average of lightgbm, xgboost, catboost"] = model.regression_metrics(y[te], avg).to_dict()

    out = pd.DataFrame(rows).T.sort_values("MAE")
    out.to_csv(ROOT / "reports" / "model_benchmark_extra.csv")
    print(out.round(3))


if __name__ == "__main__":
    main()
