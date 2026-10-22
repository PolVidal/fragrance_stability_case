"""Unit tests for the root cause helpers, on toy data where the right answer is known."""
import numpy as np
import pandas as pd
import pytest

from fragrance import rootcause


def test_mean_ci_contains_mean_and_ignores_nan():
    out = rootcause.mean_ci(pd.Series([1.0, 2.0, 3.0, np.nan]))
    assert out["mean"] == pytest.approx(2.0)
    assert out["n"] == 3
    assert out["ci_low"] < 2.0 < out["ci_high"]


def test_control_limits_use_baseline_only():
    series = pd.Series([10.0, 12.0, 10.0, 12.0, 50.0])  # last point is the "event"
    baseline = pd.Series([True, True, True, True, False])
    lim = rootcause.control_limits(series, baseline, k=3)
    assert lim["centre"] == pytest.approx(11.0)
    assert lim["ucl"] < 50.0  # the event point falls outside the limits


@pytest.fixture
def toy():
    """Two groups, before/after. Both rise by +1 after (a plant-wide change);
    the treated group additionally drops by 3 (the event)."""
    rng = np.random.default_rng(0)
    n = 2000
    treated = np.repeat([True, False], n)
    after = np.tile(np.repeat([False, True], n // 2), 2)
    y = 20 + 1.0 * after - 3.0 * (treated & after) + rng.normal(0, 0.5, 2 * n)
    return pd.DataFrame({"y": y, "treated": treated, "after": after})


def test_diff_in_diff_recovers_planted_effect(toy):
    dd = rootcause.diff_in_diff(toy, "y", toy["treated"], ~toy["treated"], toy["after"])
    assert dd["effect"] == pytest.approx(-3.0, abs=0.1)
    assert dd["ci_low"] < -3.0 < dd["ci_high"]


def test_before_after_shows_control_change(toy):
    t = rootcause.before_after(toy, "y", {"treated": toy["treated"], "control": ~toy["treated"]}, toy["after"])
    assert t.loc["control", "change"] == pytest.approx(1.0, abs=0.1)   # plant-wide change only
    assert t.loc["treated", "change"] == pytest.approx(-2.0, abs=0.1)  # +1 plant-wide, -3 event


def test_binned_curve_and_shape_detect_a_peak():
    x = pd.Series(np.linspace(0, 1, 1000))
    df = pd.DataFrame({"x": x, "y": -(x - 0.5) ** 2})  # optimum in the middle
    curve = rootcause.binned_curve(df, "x", "y", bins=10)
    shape = rootcause.effect_shape(curve)
    assert len(curve) == 10
    assert shape["shape"] == "peaked"
    assert 0.4 < shape["best_at"] < 0.6


def test_effect_shape_monotonic():
    curve = pd.DataFrame({"mean": [1.0, 2.0, 3.0]}, index=[0.1, 0.2, 0.3])
    assert rootcause.effect_shape(curve)["shape"] == "increasing"


def test_additivity_gap_is_zero_for_additive_table():
    table = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]])   # row effect + column effect only
    assert rootcause.additivity_gap(table) == pytest.approx(0.0)
    assert rootcause.additivity_gap(pd.DataFrame([[1.0, 2.0], [3.0, 9.0]])) > 0


def test_tree_rules_cover_every_batch_once():
    from sklearn.tree import DecisionTreeRegressor

    X = pd.DataFrame({"a": np.arange(100.0)})
    y = pd.Series(np.where(X["a"] < 50, 18.0, 24.0))
    tree = DecisionTreeRegressor(max_depth=1).fit(X, y)
    rules = rootcause.tree_rules(tree, ["a"], X, y)
    assert rules["batches"].sum() == 100
    assert rules.loc[0, "mean_stability"] == pytest.approx(18.0)  # sorted worst first
    assert rules.loc[0, "share_below_20"] == pytest.approx(1.0)
    assert "a ≤" in rules.loc[0, "rule"]


def test_eta_squared_extremes():
    g = pd.Series(["a", "a", "b", "b"])
    assert rootcause.eta_squared(pd.Series([1.0, 1.0, 5.0, 5.0]), g) == pytest.approx(1.0)  # all between groups
    assert rootcause.eta_squared(pd.Series([1.0, 5.0, 1.0, 5.0]), g) == pytest.approx(0.0)  # all within groups


def test_gap_breakdown_explains_a_gap_caused_by_a_feature():
    df = pd.DataFrame({"plant": ["A"] * 2 + ["B"] * 2, "x": [1.0, 1.0, 3.0, 3.0]})
    df["y"] = 10 + 2 * df["x"]          # plant B is better only because it runs at higher x
    out = rootcause.gap_breakdown(df, pd.Series({"x": 2.0}), ["x"], "plant", "A", "y")
    assert out.loc["B", "actual_gap"] == pytest.approx(4.0)
    assert out.loc["B", "explained"] == pytest.approx(4.0)
    assert out.loc["B", "unexplained"] == pytest.approx(0.0)


def test_window_summary_reports_reference_row_and_shares():
    y = pd.Series([18.0, 19.0, 22.0, 24.0])
    out = rootcause.window_summary(y, {"good": pd.Series([False, False, True, True])})
    assert out.loc["all batches", "share_below_20"] == pytest.approx(0.5)
    assert out.loc["good", "share_below_20"] == pytest.approx(0.0)
    assert out.loc["good", "share_of_production"] == pytest.approx(0.5)
