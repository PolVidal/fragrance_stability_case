"""Predictive model helpers (phase 4): splits, metrics, baselines and candidate pipelines.

Protocol (fixed before any model was run):
- one row = one batch; inputs = the 7 sensors (variant without density for predictions made
  before the separation step); plant, reactor and date are not inputs;
- primary evaluation on a time split, every fitted step learned on training data only;
- records with every sensor missing are excluded from training and evaluated separately.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fragrance import config

TRAIN_END = pd.Timestamp("2023-06-09")   # train: before this day; test: from this day
CALIB_START = pd.Timestamp("2023-05-30")  # last 10 training days calibrate the intervals
LOW_THRESHOLD = 20.0                      # Technical grade: stability below 20 months
SEED = 42

FEATURES = config.CLEAN_SENSOR_COLS
FEATURES_NO_DENSITY = [c for c in FEATURES if c != "density_g_cm3"]


def time_split(df: pd.DataFrame, time: str = "event_time") -> dict[str, pd.Series]:
    """WHAT: boolean masks `train`, `fit` (train before the calibration window), `calib`
    (last 10 training days) and `test` (from TRAIN_END), on calendar time.
    WHY: the honest test of a production model is the future: train on the past, test on later
    batches it could not have seen. The calibration window is the most recent past, held out
    from fitting so the prediction intervals are calibrated on unseen batches.
    """
    t = df[time]
    train = t < TRAIN_END
    return {"train": train, "fit": train & (t < CALIB_START), "calib": train & (t >= CALIB_START),
            "test": ~train}


def random_split(df: pd.DataFrame, test_size: float = 0.25, seed: int = SEED) -> dict[str, pd.Series]:
    """WHAT: boolean masks `train` and `test` from a random split of batches.
    WHY: secondary comparison only. One row is one batch, so it cannot leak a batch, but it mixes
    past and future and is expected to look slightly better than the time split.
    """
    rng = np.random.default_rng(seed)
    test = pd.Series(rng.random(len(df)) < test_size, index=df.index)
    return {"train": ~test, "test": test}


def regression_metrics(y_true: pd.Series, y_pred, low_threshold: float = LOW_THRESHOLD) -> pd.Series:
    """WHAT: MAE overall, MAE on batches whose *true* stability is below `low_threshold`
    (Technical grade), RMSE and R², plus the number of batches.
    WHY: MAE is in months and easy to explain; the low-zone MAE checks the model where the
    business cares most, since a model can look good on average while missing the bad batches.
    """
    y_true = pd.Series(y_true).reset_index(drop=True)
    err = pd.Series(np.asarray(y_pred, dtype=float)) - y_true
    low = y_true < low_threshold
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return pd.Series({
        "MAE": err.abs().mean(),
        "MAE_technical": err[low].abs().mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "R2": 1 - (err ** 2).sum() / ss_tot,
        "n": len(y_true),
    })


class MeanBaseline:
    """Predicts the training mean for every batch: the "know nothing" reference."""

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.mean_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_)


class GroupMeanBaseline:
    """Predicts the training mean of the batch's group (e.g. reactor); unseen groups get the
    overall training mean. The "naive" baseline a plant would use without any sensor."""

    def __init__(self, group: str = "reactor"):
        self.group = group

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.means_ = y.groupby(X[self.group]).mean()
        self.overall_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.group].map(self.means_).fillna(self.overall_).to_numpy()


# --------------------------------------------------------------------------- #
# Candidate pipelines (4.2). Every step is inside the pipeline, so it is fitted on
# training data only.
# --------------------------------------------------------------------------- #
class WindowFeatures:
    """Turns raw sensors into the phase-3 features for a linear model: pH and pressure become
    their distance from the best value, **learned on the training data** (the value of the
    best-stability decile), so nothing about the test set leaks in. Other sensors pass through."""

    peaked = ("ph_level", "vessel_pressure_bar")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.best_ = {}
        for col in self.peaked:
            if col in X:
                d = pd.DataFrame({"x": X[col], "y": y}).dropna()
                bins = pd.qcut(d["x"], 10, duplicates="drop")
                means = d.groupby(bins, observed=True)["y"].mean()
                self.best_[col] = float(d.groupby(bins, observed=True)["x"].median()[means.idxmax()])
        self.columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, best in self.best_.items():
            X[col] = (X[col] - best).abs()
        return X.rename(columns={c: f"{c}_distance" for c in self.best_})

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_distance" if c in self.best_ else c for c in self.columns_])


def make_candidates(seed: int = SEED) -> dict:
    """WHAT: the candidate model families of the protocol, as sklearn pipelines.
    WHY: one place where every candidate is defined, so the comparison is like-for-like.
    - linear: phase-3 features (distance from best pH / pressure), median imputation with
      missing-value indicators, scaling, ordinary least squares;
    - splines: an additive spline model (one smooth curve per sensor, summed), GAM-like;
    - random forest: median imputation with indicators, limited leaf size;
    - lightgbm: gradient boosting with native handling of missing values.
    """
    import lightgbm as lgb
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer, StandardScaler

    class _Window(WindowFeatures, BaseEstimator, TransformerMixin):
        pass

    impute = lambda: SimpleImputer(strategy="median", add_indicator=True)
    return {
        "linear": make_pipeline(_Window(), impute(), StandardScaler(), LinearRegression()),
        "splines": make_pipeline(impute(), SplineTransformer(n_knots=8, degree=3), Ridge(alpha=1.0)),
        "random forest": make_pipeline(impute(), RandomForestRegressor(
            n_estimators=200, min_samples_leaf=50, max_features=0.6, n_jobs=-1, random_state=seed)),
        "lightgbm": lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31,
                                      min_child_samples=100, random_state=seed, verbose=-1),
    }


def paired_error_difference(y_true: pd.Series, pred_a, pred_b) -> pd.Series:
    """WHAT: mean of |error of A| − |error of B| over the same batches, with a 95% CI.
    WHY: two models are scored on the *same* test batches, so the fair question is whether A's
    error is lower batch by batch; a paired comparison is far more precise than comparing two
    MAEs with separate intervals. Negative = A is better.
    """
    y = np.asarray(y_true, dtype=float)
    d = np.abs(np.asarray(pred_a) - y) - np.abs(np.asarray(pred_b) - y)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return pd.Series({"diff": d.mean(), "ci_low": d.mean() - 1.96 * se, "ci_high": d.mean() + 1.96 * se})


# --------------------------------------------------------------------------- #
# 4.3 Robustness checks and error analysis
# --------------------------------------------------------------------------- #
def walk_forward_splits(df: pd.DataFrame, test_starts: list[str], test_days: int = 10,
                        time: str = "event_time") -> list[dict[str, pd.Series]]:
    """WHAT: one (train, test) pair of masks per start date: train = everything before the start,
    test = the next `test_days` days (an expanding training window).
    WHY: a single test period is one draw; repeating "train on the past, test on the next days"
    several times shows whether errors and the ranking of models are stable over time. All folds
    lie before TRAIN_END, so the final test period stays untouched.
    """
    out = []
    for start in pd.to_datetime(test_starts):
        end = start + pd.Timedelta(days=test_days)
        out.append({"train": df[time] < start, "test": (df[time] >= start) & (df[time] < end)})
    return out


def technical_detection(y_true: pd.Series, y_pred, threshold: float = LOW_THRESHOLD) -> pd.Series:
    """WHAT: the regression turned into a yes/no decision, "Technical grade (< threshold)?":
    confusion counts, accuracy, precision, recall, F1, plus ROC-AUC and PR-AUC computed from the
    predicted stability as a risk score (lower prediction = higher risk).
    WHY: the business question is "will the model catch the bad batches?". Recall = share of truly
    Technical batches that are flagged; precision = share of flagged batches that really are
    Technical. AUCs judge the ranking at every possible threshold, not just at 20.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    truth = np.asarray(y_true) < threshold
    pred = np.asarray(y_pred, dtype=float)
    flag = pred < threshold
    tp, fp = int((flag & truth).sum()), int((flag & ~truth).sum())
    fn, tn = int((~flag & truth).sum()), int((~flag & ~truth).sum())
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    return pd.Series({
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "accuracy": (tp + tn) / len(truth), "precision": precision, "recall": recall,
        "F1": 2 * precision * recall / (precision + recall) if precision + recall else np.nan,
        "ROC_AUC": roc_auc_score(truth, -pred), "PR_AUC": average_precision_score(truth, -pred),
        "share_truly_technical": truth.mean(),
    })


def mae_by_group(y_true: pd.Series, y_pred, groups: pd.Series) -> pd.DataFrame:
    """WHAT: MAE, mean error (bias: + = over-predicts) and count per group.
    WHY: an average MAE can hide a group where the model is systematically wrong (a plant, a
    period, batches with missing sensors)."""
    err = pd.Series(np.asarray(y_pred, dtype=float) - np.asarray(y_true), index=groups.index)
    return pd.DataFrame({"MAE": err.abs().groupby(groups).mean(), "bias": err.groupby(groups).mean(),
                         "n": err.groupby(groups).size()})


# --------------------------------------------------------------------------- #
# 4.4 Prediction intervals and out-of-distribution flag
# --------------------------------------------------------------------------- #
def conformal_quantile(scores, alpha: float = 0.10) -> float:
    """WHAT: the finite-sample conformal quantile of calibration scores: the ⌈(n+1)(1−α)⌉-th
    smallest score.
    WHY: with this exact rank (not the plain 90th percentile) the guarantee "at least 1 − α of
    new batches fall inside the interval" holds for any sample size, assuming new batches
    behave like the calibration batches.
    """
    s = np.sort(np.asarray(scores, dtype=float))
    k = int(np.ceil((len(s) + 1) * (1 - alpha)))
    return float(s[min(k, len(s)) - 1])


class ConformalQuantileInterval:
    """Conformalized quantile regression (CQR, Romano et al. 2019).

    1. Two LightGBM quantile models learn a low (5%) and a high (95%) value from the sensors,
       so the interval is naturally wider for harder batches (e.g. missing sensors).
    2. On the calibration batches, the score is how far each true value falls outside its
       interval (negative if inside); its conformal quantile q widens (or narrows) every
       interval just enough that 90% of calibration batches would be covered.
    Interval for a new batch: [low − q, high + q].
    """

    def __init__(self, alpha: float = 0.10, seed: int = SEED, **lgb_params):
        self.alpha, self.seed = alpha, seed
        self.lgb_params = {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 31,
                           "min_child_samples": 100, **lgb_params}

    def fit(self, X_fit, y_fit, X_calib, y_calib):
        import lightgbm as lgb

        self.low_ = lgb.LGBMRegressor(objective="quantile", alpha=self.alpha / 2, random_state=self.seed,
                                      verbose=-1, **self.lgb_params).fit(X_fit, y_fit)
        self.high_ = lgb.LGBMRegressor(objective="quantile", alpha=1 - self.alpha / 2, random_state=self.seed,
                                       verbose=-1, **self.lgb_params).fit(X_fit, y_fit)
        lo, hi = self.low_.predict(X_calib), self.high_.predict(X_calib)
        y = np.asarray(y_calib, dtype=float)
        self.q_ = conformal_quantile(np.maximum(lo - y, y - hi), self.alpha)
        return self

    def predict_interval(self, X) -> tuple[np.ndarray, np.ndarray]:
        return self.low_.predict(X) - self.q_, self.high_.predict(X) + self.q_


def interval_coverage(y_true, low, high, groups: pd.Series | None = None) -> pd.DataFrame:
    """WHAT: share of batches whose true value lies inside [low, high] and mean interval width,
    overall or per group.
    WHY: a 90% interval is only honest if it contains the truth about 90% of the time, and
    should do so for every kind of batch, not just on average."""
    y = np.asarray(y_true, dtype=float)
    d = pd.DataFrame({"inside": (y >= low) & (y <= high), "width": np.asarray(high) - np.asarray(low)},
                     index=groups.index if groups is not None else None)
    if groups is None:
        return d.agg({"inside": "mean", "width": "mean"}).to_frame("all").T.rename(columns={"inside": "coverage"})
    return d.groupby(groups).agg(coverage=("inside", "mean"), width=("width", "mean"), n=("inside", "size"))


def risk_tier(low, high, threshold: float = LOW_THRESHOLD) -> np.ndarray:
    """WHAT: a traffic light per batch from its interval: "red" if even the upper bound is below
    the threshold (almost surely Technical), "green" if even the lower bound is above it (almost
    surely fine), "amber" if the interval straddles it (uncertain: check in the lab).
    WHY: turns the interval into the yes/no/maybe decision the Quality team needs."""
    low, high = np.asarray(low), np.asarray(high)
    return np.where(high < threshold, "red", np.where(low >= threshold, "green", "amber"))


class RangeFlag:
    """Out-of-distribution flag: a batch is flagged when any sensor lies outside the range seen
    in training (between the 0.1% and 99.9% quantiles). Missing values are never flagged.
    WHY: a tree model cannot predict beyond the values it was trained on; this says "the model
    has never seen conditions like these, trust its prediction less"."""

    def __init__(self, lower: float = 0.001, upper: float = 0.999):
        self.lower, self.upper = lower, upper

    def fit(self, X: pd.DataFrame):
        self.low_, self.high_ = X.quantile(self.lower), X.quantile(self.upper)
        return self

    def flag(self, X: pd.DataFrame) -> np.ndarray:
        return ((X < self.low_) | (X > self.high_)).any(axis=1).to_numpy()

    def which(self, row: pd.Series) -> list[str]:
        """Names of the sensors of one batch that lie outside the training range."""
        r = row[self.low_.index]
        return list(r.index[((r < self.low_) | (r > self.high_)).fillna(False)])


# --------------------------------------------------------------------------- #
# 4.5 The final package: prediction + interval + traffic light + flag + explanation
# --------------------------------------------------------------------------- #
UNITS = {"reaction_temperature_celsius": "°C", "vessel_pressure_bar": "bar", "refractive_index": "",
         "ph_level": "", "mixing_torque_nm": "N·m", "mass_flow_rate_kg_h": "kg/h", "density_g_cm3": "g/cm³"}
DECIMALS = {"refractive_index": 4, "density_g_cm3": 3}   # others: 2
LABELS = {"reaction_temperature_celsius": "temperature", "vessel_pressure_bar": "pressure",
          "refractive_index": "refractive index", "ph_level": "pH", "mixing_torque_nm": "mixing torque",
          "mass_flow_rate_kg_h": "mass flow", "density_g_cm3": "density"}


class StabilityPredictor:
    """Everything the Quality team gets for a batch, in one object:
    - `prediction`: LightGBM point prediction (months), from the sensors;
    - `low`, `high`: 90% interval (CQR, calibrated on unseen batches);
    - `tier`: red / amber / green from the interval and the 20-month Technical limit;
    - `out_of_range`: True if a sensor lies outside what the model was trained on;
    - `no_sensor_data`: batches without any sensor reading get their reactor's training average
      and a wider conformal interval (the sensors cannot say anything about them).
    `explain` gives each sensor's contribution in months (SHAP) for one batch.
    """

    def __init__(self, features: list[str] = FEATURES_NO_DENSITY, alpha: float = 0.10, seed: int = SEED):
        self.features, self.alpha, self.seed = list(features), alpha, seed

    def fit(self, fit_df: pd.DataFrame, calib_df: pd.DataFrame, target: str = config.TARGET,
            group: str = "reactor"):
        """`fit_df` trains the models; `calib_df` (later, unseen batches) calibrates the intervals.
        Rows with no sensor data are ignored except for the reactor fallback."""
        has = lambda d: d[self.features].notna().any(axis=1)
        f, c = fit_df[has(fit_df)], calib_df[has(calib_df)]
        self.point_ = make_candidates(self.seed)["lightgbm"].fit(f[self.features], f[target])
        self.interval_ = ConformalQuantileInterval(self.alpha, self.seed).fit(
            f[self.features], f[target], c[self.features], c[target])
        self.range_ = RangeFlag().fit(f[self.features])
        self.group = group
        self.fallback_means_ = fit_df.groupby(group)[target].mean()
        self.fallback_overall_ = float(fit_df[target].mean())
        fallback_calib = calib_df[group].map(self.fallback_means_).fillna(self.fallback_overall_)
        self.fallback_q_ = conformal_quantile(np.abs(calib_df[target] - fallback_calib), self.alpha)
        return self

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """One row per batch: prediction, low, high, tier, out_of_range, no_sensor_data."""
        X = df[self.features]
        empty = X.isna().all(axis=1).to_numpy()
        pred = self.point_.predict(X)
        low, high = self.interval_.predict_interval(X)
        if empty.any():
            fb = df.loc[empty, self.group].map(self.fallback_means_).fillna(self.fallback_overall_).to_numpy()
            pred[empty], low[empty], high[empty] = fb, fb - self.fallback_q_, fb + self.fallback_q_
        out = pd.DataFrame({"prediction": pred, "low": low, "high": high}, index=df.index)
        out["tier"] = risk_tier(out["low"], out["high"])
        out["out_of_range"] = self.range_.flag(X) & ~empty
        out["no_sensor_data"] = empty
        return out

    def explain(self, df: pd.DataFrame) -> pd.DataFrame:
        """WHAT: SHAP contribution (months) of each sensor for each batch, plus the base value
        (average prediction) so that base + contributions = prediction.
        WHY: says *why* a batch is predicted low: which conditions pulled it down, by how much."""
        import shap

        explainer = shap.TreeExplainer(self.point_)
        values = explainer.shap_values(df[self.features])
        out = pd.DataFrame(values, columns=self.features, index=df.index)
        out.insert(0, "base", float(np.ravel(explainer.expected_value)[0]))
        return out

    def explain_text(self, row: pd.Series, result: pd.Series, contributions: pd.Series, top: int = 3) -> str:
        """A one-paragraph explanation for one batch, in plain words."""
        if result["no_sensor_data"]:
            return (f"No sensor data: {result['prediction']:.1f} months is the reactor's usual level "
                    f"(90% interval {result['low']:.1f}–{result['high']:.1f}), tier {result['tier']}. Check in the lab.")
        contrib = contributions[self.features].sort_values(key=np.abs, ascending=False).head(top)
        reasons = []
        for feat, c in contrib.items():
            value = row[feat]
            d = DECIMALS.get(feat, 2)
            shown = "missing" if pd.isna(value) else f"{value:.{d}f} {UNITS[feat]}".strip()
            reasons.append(f"{LABELS[feat]} {shown} ({c:+.1f} months)")
        text = (f"Predicted {result['prediction']:.1f} months (90% interval {result['low']:.1f}–{result['high']:.1f}), "
                f"tier {result['tier']}. Main reasons vs the average batch ({contributions['base']:.1f}): "
                + "; ".join(reasons) + ".")
        if result["out_of_range"]:
            d = lambda f: DECIMALS.get(f, 2)
            outside = ", ".join(f"{LABELS[f]} {row[f]:.{d(f)}f} (trained on {self.range_.low_[f]:.{d(f)}f}–"
                                f"{self.range_.high_[f]:.{d(f)}f})" for f in self.range_.which(row))
            text += f" Warning: {outside} is outside what the model has seen; trust the prediction less."
        return text

    def save(self, path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "StabilityPredictor":
        import joblib

        return joblib.load(path)
