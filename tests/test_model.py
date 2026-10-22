"""Unit tests for the model helpers: splits, metrics and baselines on toy data."""
import numpy as np
import pandas as pd
import pytest

from fragrance import model


def test_time_split_is_ordered_and_disjoint():
    df = pd.DataFrame({"event_time": pd.date_range("2023-04-01", "2023-06-29", freq="6h")})
    s = model.time_split(df)
    assert not (s["train"] & s["test"]).any()
    assert (s["train"] | s["test"]).all()
    assert df.loc[s["train"], "event_time"].max() < df.loc[s["test"], "event_time"].min()
    assert ((s["fit"] | s["calib"]) == s["train"]).all() and not (s["fit"] & s["calib"]).any()
    assert df.loc[s["fit"], "event_time"].max() < df.loc[s["calib"], "event_time"].min()


def test_random_split_is_reproducible():
    df = pd.DataFrame({"x": range(1000)})
    a, b = model.random_split(df), model.random_split(df)
    assert (a["test"] == b["test"]).all()
    assert 0.2 < a["test"].mean() < 0.3


def test_regression_metrics_known_values():
    y = pd.Series([18.0, 19.0, 22.0, 24.0])
    pred = [19.0, 19.0, 22.0, 26.0]                 # errors 1, 0, 0, 2
    m = model.regression_metrics(y, pred)
    assert m["MAE"] == pytest.approx(0.75)
    assert m["MAE_technical"] == pytest.approx(0.5)  # only the two batches below 20
    assert m["RMSE"] == pytest.approx(np.sqrt(5 / 4))


def test_group_mean_baseline_falls_back_for_unseen_group():
    X = pd.DataFrame({"reactor": ["a", "a", "b"]})
    y = pd.Series([20.0, 22.0, 30.0])
    b = model.GroupMeanBaseline().fit(X, y)
    pred = b.predict(pd.DataFrame({"reactor": ["a", "c"]}))
    assert pred[0] == pytest.approx(21.0)
    assert pred[1] == pytest.approx(24.0)          # unseen reactor -> overall mean


def test_window_features_learn_best_value_on_training_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"ph_level": rng.uniform(5, 6, 2000), "other": rng.normal(size=2000)})
    y = pd.Series(25 - 10 * (X["ph_level"] - 5.4).abs())      # best pH = 5.4
    w = model.WindowFeatures().fit(X, y)
    assert w.best_["ph_level"] == pytest.approx(5.4, abs=0.06)
    out = w.transform(X)
    assert "ph_level_distance" in out.columns and (out["ph_level_distance"] >= 0).all()


def test_paired_error_difference_sign():
    y = pd.Series([20.0, 21.0, 22.0])
    better, worse = [20.1, 21.1, 22.1], [21.0, 22.0, 23.0]
    d = model.paired_error_difference(y, better, worse)
    assert d["diff"] == pytest.approx(-0.9)


def test_walk_forward_folds_are_ordered_and_before_test_period():
    df = pd.DataFrame({"event_time": pd.date_range("2023-04-01", "2023-06-29", freq="12h")})
    folds = model.walk_forward_splits(df, ["2023-05-10", "2023-05-20"], test_days=10)
    for f in folds:
        assert df.loc[f["train"], "event_time"].max() < df.loc[f["test"], "event_time"].min()
        assert df.loc[f["test"], "event_time"].max() < model.TRAIN_END
    assert folds[1]["train"].sum() > folds[0]["train"].sum()   # expanding window


def test_technical_detection_counts():
    y = pd.Series([18.0, 19.0, 21.0, 25.0])
    pred = [19.5, 20.5, 19.0, 24.0]      # catches 1 of 2 Technical, 1 false alarm
    d = model.technical_detection(y, pred)
    assert (d["TP"], d["FN"], d["FP"], d["TN"]) == (1, 1, 1, 1)
    assert d["precision"] == pytest.approx(0.5) and d["recall"] == pytest.approx(0.5)
    assert 0 <= d["ROC_AUC"] <= 1


def test_mae_by_group_bias_sign():
    y = pd.Series([20.0, 20.0, 22.0])
    out = model.mae_by_group(y, [21.0, 21.0, 21.0], pd.Series(["a", "a", "b"]))
    assert out.loc["a", "bias"] == pytest.approx(1.0)     # over-predicts
    assert out.loc["b", "bias"] == pytest.approx(-1.0)


def test_conformal_quantile_rank():
    scores = np.arange(1, 10)                         # n = 9, alpha 0.1 -> rank ceil(10*0.9) = 9
    assert model.conformal_quantile(scores, 0.10) == 9
    assert model.conformal_quantile(np.arange(1, 100), 0.10) == 90


def test_cqr_reaches_nominal_coverage_on_synthetic_data():
    rng = np.random.default_rng(0)
    n = 6000
    X = pd.DataFrame({"x": rng.uniform(0, 1, n)})
    y = pd.Series(20 + 3 * X["x"] + rng.normal(0, 0.5 + X["x"], n))   # noise grows with x
    fit, cal, test = slice(0, 3000), slice(3000, 4500), slice(4500, n)
    m = model.ConformalQuantileInterval(n_estimators=200).fit(X[fit], y[fit], X[cal], y[cal])
    lo, hi = m.predict_interval(X[test])
    cov = model.interval_coverage(y[test], lo, hi)
    assert 0.86 <= cov.loc["all", "coverage"] <= 0.94
    width = pd.Series(hi - lo)
    assert width[X["x"][test].to_numpy() > 0.8].mean() > width[X["x"][test].to_numpy() < 0.2].mean()


def test_risk_tier():
    out = model.risk_tier([18.0, 19.0, 20.5], [19.5, 21.0, 23.0])
    assert list(out) == ["red", "amber", "green"]


def test_range_flag():
    train = pd.DataFrame({"a": np.linspace(0, 10, 1001)})
    f = model.RangeFlag().fit(train)
    assert list(f.flag(pd.DataFrame({"a": [5.0, -3.0, np.nan]}))) == [False, True, False]


@pytest.fixture
def toy_batches():
    rng = np.random.default_rng(1)
    n = 3000
    df = pd.DataFrame({c: rng.normal(size=n) for c in model.FEATURES_NO_DENSITY})
    df["reactor"] = rng.choice(["Plant-01/R01", "Plant-02/R01"], n)
    df[model.config.TARGET] = 21 + 1.5 * df["mixing_torque_nm"] - 0.8 * df["reaction_temperature_celsius"] + rng.normal(0, 0.5, n)
    df.loc[df.index[-5:], model.FEATURES_NO_DENSITY] = np.nan          # a few batches without sensor data
    return df


def test_stability_predictor_outputs_and_explanations(toy_batches):
    fit_df, calib_df, new = toy_batches.iloc[:2000], toy_batches.iloc[2000:2600], toy_batches.iloc[2600:]
    p = model.StabilityPredictor().fit(fit_df, calib_df)
    out = p.predict_frame(new)
    assert list(out.columns) == ["prediction", "low", "high", "tier", "out_of_range", "no_sensor_data"]
    assert (out["low"] <= out["prediction"]).mean() > 0.95 and (out["high"] >= out["prediction"]).mean() > 0.95
    assert out["no_sensor_data"].sum() == 5 and set(out["tier"]) <= {"red", "amber", "green"}
    with_sensors = new[~out["no_sensor_data"]].head(20)
    contrib = p.explain(with_sensors)
    rebuilt = contrib.sum(axis=1)                                     # base + contributions = prediction
    np.testing.assert_allclose(rebuilt, out.loc[with_sensors.index, "prediction"], atol=1e-4)
    text = p.explain_text(with_sensors.iloc[0], out.loc[with_sensors.index[0]], contrib.iloc[0])
    assert "months" in text and "tier" in text


def test_range_flag_names_the_sensor():
    f = model.RangeFlag().fit(pd.DataFrame({"a": np.linspace(0, 10, 1001), "b": np.linspace(0, 1, 1001)}))
    assert f.which(pd.Series({"a": 50.0, "b": 0.5})) == ["a"]
