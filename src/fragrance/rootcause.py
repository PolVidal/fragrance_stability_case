"""Root cause analysis helpers (phase 3). Every function reads the clean table and returns a
summary; nothing is modified. Methods are kept standard on purpose: group means with
confidence intervals, control charts, and comparisons against an unaffected control group.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Z_95 = 1.96  # normal quantile for a 95% confidence interval


def mean_ci(x: pd.Series) -> pd.Series:
    """WHAT: mean, 95% confidence interval and count of `x` (NaN ignored).
    WHY: a mean without its uncertainty cannot be compared with another mean.
    HOW: normal approximation, mean ± 1.96 · std / √n. Fine here: groups have hundreds of
    batches or more, and consecutive batches are practically independent (audit, phase 2).
    """
    x = x.dropna()
    se = x.std(ddof=1) / np.sqrt(len(x))
    m = x.mean()
    return pd.Series({"mean": m, "ci_low": m - Z_95 * se, "ci_high": m + Z_95 * se, "n": len(x)})


def diff_ci(a: pd.Series, b: pd.Series) -> pd.Series:
    """WHAT: difference of means a − b with its 95% confidence interval.
    WHY: "how much lower, and how sure are we" in one line.
    HOW: the standard errors of two independent means add in quadrature: √(se_a² + se_b²).
    """
    a, b = a.dropna(), b.dropna()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    d = a.mean() - b.mean()
    return pd.Series({"diff": d, "ci_low": d - Z_95 * se, "ci_high": d + Z_95 * se})


def daily_mean(df: pd.DataFrame, value: str, by: str, time: str = "event_time") -> pd.DataFrame:
    """WHAT: mean of `value` per calendar day (rows) and group (columns).
    WHY: one point per day smooths batch-to-batch noise and makes shifts visible.
    """
    return df.groupby([df[time].dt.normalize(), by])[value].mean().unstack()


def control_limits(series: pd.Series, baseline: pd.Series, k: float = 3.0) -> pd.Series:
    """WHAT: centre line and lower/upper control limits from a baseline period.
    WHY: a Shewhart control chart, the standard tool of manufacturing quality: points outside
    centre ± k·std of the baseline are very unlikely (~0.3% for k = 3) if nothing changed.
    `series` is one value per period (e.g. daily mean); `baseline` is a boolean mask.
    """
    ref = series[baseline]
    centre, sd = ref.mean(), ref.std(ddof=1)
    return pd.Series({"centre": centre, "lcl": centre - k * sd, "ucl": centre + k * sd})


def before_after(df: pd.DataFrame, value: str, groups: dict[str, pd.Series],
                 after: pd.Series) -> pd.DataFrame:
    """WHAT: for each named group (boolean mask), the mean of `value` before and after a split
    (boolean mask `after`), and the change with its 95% CI.
    WHY: compares an affected group with a control group around the same date: if only the
    affected group moves, the change is tied to the event and not to a plant-wide or seasonal
    effect (the logic of a difference-in-differences comparison).
    """
    rows = {}
    for name, mask in groups.items():
        pre, post = df.loc[mask & ~after, value], df.loc[mask & after, value]
        change = diff_ci(post, pre)
        rows[name] = {"before": pre.mean(), "after": post.mean(), "change": change["diff"],
                      "ci_low": change["ci_low"], "ci_high": change["ci_high"],
                      "n_before": pre.notna().sum(), "n_after": post.notna().sum()}
    return pd.DataFrame(rows).T


def diff_in_diff(df: pd.DataFrame, value: str, treated: pd.Series, control: pd.Series,
                 after: pd.Series) -> pd.Series:
    """WHAT: (treated after − before) − (control after − before), with its 95% CI.
    WHY: the change attributable to the event, net of anything that moved both groups.
    HOW: four independent group means; their standard errors add in quadrature.
    """
    cells = [df.loc[m, value].dropna() for m in
             (treated & after, treated & ~after, control & after, control & ~after)]
    est = cells[0].mean() - cells[1].mean() - (cells[2].mean() - cells[3].mean())
    se = np.sqrt(sum(c.var(ddof=1) / len(c) for c in cells))
    return pd.Series({"effect": est, "ci_low": est - Z_95 * se, "ci_high": est + Z_95 * se})


def sensor_shift(df: pd.DataFrame, sensors: list[str], treated: pd.Series, control: pd.Series,
                 after: pd.Series) -> pd.DataFrame:
    """WHAT: for each sensor, the event effect (difference-in-differences) in the sensor's own
    unit and in standard deviations of the control group before the event.
    WHY: the event's "fingerprint": which measured conditions changed with stability, and by
    how much relative to their normal variation.
    """
    out = {}
    for s in sensors:
        dd = diff_in_diff(df, s, treated, control, after)
        sd = df.loc[control & ~after, s].std(ddof=1)
        out[s] = {**dd.to_dict(), "effect_in_sd": dd["effect"] / sd}
    return pd.DataFrame(out).T


# --------------------------------------------------------------------------- #
# 3.2 Process window
# --------------------------------------------------------------------------- #
def binned_curve(df: pd.DataFrame, x: str, y: str, bins: int = 10) -> pd.DataFrame:
    """WHAT: mean of `y` (with 95% CI and count) in each of `bins` equal-count bins of `x`,
    indexed by the median of `x` in the bin.
    WHY: shows the *shape* of an effect (rising, falling, peaked) without assuming a straight
    line. Equal-count bins (deciles) keep every point equally precise.
    """
    d = df[[x, y]].dropna()
    groups = d.groupby(pd.qcut(d[x], bins, duplicates="drop"), observed=True)
    out = groups[y].apply(mean_ci).unstack()
    out.index = groups[x].median().values
    out.index.name = x
    return out


def effect_shape(curve: pd.DataFrame) -> pd.Series:
    """WHAT: summary of a binned curve: swing (best bin mean − worst bin mean, in months),
    the x value of the best bin, and the shape: "increasing", "decreasing" or "peaked"
    (best bin strictly inside the range).
    WHY: one comparable line per sensor. A peaked effect means an optimum window: both too
    low and too high are bad, which a simple comparison of means cannot see.
    """
    m = curve["mean"].to_numpy()
    best = int(np.argmax(m))
    if 0 < best < len(m) - 1:
        shape = "peaked"
    else:
        shape = "increasing" if m[-1] > m[0] else "decreasing"
    return pd.Series({"swing": m.max() - m.min(), "best_at": curve.index[best], "shape": shape})


def group_test(a: pd.Series, b: pd.Series) -> pd.Series:
    """WHAT: difference of means a − b with 95% CI, plus two p-values: Welch's t-test
    (compares means, does not assume equal variances) and Mann-Whitney U (compares ranks,
    does not assume normality).
    WHY: for small subgroups, where the uncertainty is real and a test is informative.
    """
    from scipy import stats

    a, b = a.dropna(), b.dropna()
    out = diff_ci(a, b)
    out["p_welch"] = stats.ttest_ind(a, b, equal_var=False).pvalue
    out["p_mannwhitney"] = stats.mannwhitneyu(a, b).pvalue
    out["n_a"], out["n_b"] = len(a), len(b)
    return out


def two_way_means(df: pd.DataFrame, x1: str, x2: str, y: str, bins: int = 5) -> pd.DataFrame:
    """WHAT: mean of `y` for each combination of quantile bins of `x1` (rows) and `x2`
    (columns), labelled by bin medians.
    WHY: an interaction means the effect of one variable depends on the level of the other;
    in this table it shows as rows that are not parallel.
    """
    d = df[[x1, x2, y]].dropna()
    b1, b2 = pd.qcut(d[x1], bins), pd.qcut(d[x2], bins)
    table = d.groupby([b1, b2], observed=True)[y].mean().unstack()
    table.index = d.groupby(b1, observed=True)[x1].median().round(1).values
    table.columns = d.groupby(b2, observed=True)[x2].median().round(2).values
    table.index.name, table.columns.name = x1, x2
    return table


def additivity_gap(table: pd.DataFrame) -> float:
    """WHAT: the largest difference between a two-way table of means and the best purely
    additive table (row effect + column effect).
    WHY: 0 means no interaction at all; compare it with the size of the main effects.
    """
    v = table.to_numpy()
    additive = v.mean(axis=1, keepdims=True) + v.mean(axis=0, keepdims=True) - v.mean()
    return float(np.abs(v - additive).max())


# --------------------------------------------------------------------------- #
# 3.3 Readable rules and a cross-check model
# --------------------------------------------------------------------------- #
def tree_rules(tree, feature_names: list[str], X: pd.DataFrame, y: pd.Series,
               low_threshold: float = 20.0) -> pd.DataFrame:
    """WHAT: one row per leaf of a fitted sklearn decision tree: the rule that leads to it
    (conditions joined by AND), the number of batches, their mean stability and the share
    below `low_threshold`, sorted from worst to best.
    WHY: turns the tree into a table an engineer can read and check against the plant floor.
    """
    t = tree.tree_
    paths: dict[int, dict[str, list[float]]] = {}

    def walk(node: int, bounds: dict[str, list[float]]):
        if t.children_left[node] == -1:  # leaf
            paths[node] = bounds
            return
        name, thr = feature_names[t.feature[node]], t.threshold[node]
        lo, hi = bounds.get(name, [-np.inf, np.inf])
        walk(t.children_left[node], {**bounds, name: [lo, min(hi, thr)]})
        walk(t.children_right[node], {**bounds, name: [max(lo, thr), hi]})

    def as_text(bounds: dict[str, list[float]]) -> str:
        """Conditions on the same feature merged into one range, e.g. "11.32 < torque ≤ 11.74"."""
        parts = []
        for name, (lo, hi) in bounds.items():
            if np.isfinite(lo) and np.isfinite(hi):
                parts.append(f"{lo:.2f} < {name} ≤ {hi:.2f}")
            elif np.isfinite(hi):
                parts.append(f"{name} ≤ {hi:.2f}")
            else:
                parts.append(f"{name} > {lo:.2f}")
        return " AND ".join(parts)

    walk(0, {})
    leaf = pd.Series(tree.apply(X), index=X.index)
    rows = []
    for node, bounds in paths.items():
        yy = y[leaf == node]
        rows.append({"rule": as_text(bounds), "batches": len(yy), "mean_stability": yy.mean(),
                     f"share_below_{low_threshold:g}": (yy < low_threshold).mean()})
    return pd.DataFrame(rows).sort_values("mean_stability").reset_index(drop=True)


def shap_summary(shap_values: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    """WHAT: mean absolute SHAP value per feature (average size of its contribution to a
    prediction, in months), sorted, with each feature's share of the total.
    WHY: a global importance that is in the unit of the target and adds up across features.
    """
    imp = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns).sort_values(ascending=False)
    return pd.DataFrame({"mean_abs_shap": imp, "share": imp / imp.sum()})


# --------------------------------------------------------------------------- #
# 3.4 Plants and reactors
# --------------------------------------------------------------------------- #
def eta_squared(y: pd.Series, groups: pd.Series) -> float:
    """WHAT: share of the variance of `y` explained by the group means (one-way ANOVA η²):
    between-group sum of squares / total sum of squares, between 0 and 1.
    WHY: "how much does knowing the plant (or reactor) tell us about stability?", in one number
    that is comparable across groupings, unlike an F statistic or a p-value.
    """
    d = pd.DataFrame({"y": y, "g": groups}).dropna()
    grand = d["y"].mean()
    between = ((d.groupby("g")["y"].transform("mean") - grand) ** 2).sum()
    return float(between / ((d["y"] - grand) ** 2).sum())


def gap_breakdown(df: pd.DataFrame, coefs: pd.Series, features: list[str], group: str,
                  reference: str, target: str) -> pd.DataFrame:
    """WHAT: for each group, its gap in mean `target` versus the `reference` group, split into
    the part due to each feature: coefficient × (group mean − reference mean) of that feature,
    from a linear model fitted *without* the group. Columns: one per feature, the total
    explained, the actual gap and the unexplained rest.
    WHY: answers "is plant X worse, or does it just operate differently?". If the explained total
    matches the actual gap, the plant itself adds nothing beyond its operating conditions.
    """
    means = df.groupby(group)[features + [target]].mean()
    diff = means - means.loc[reference]
    out = diff[features] * coefs[features]
    out["explained"] = out[features].sum(axis=1)
    out["actual_gap"] = diff[target]
    out["unexplained"] = out["actual_gap"] - out["explained"]
    return out


# --------------------------------------------------------------------------- #
# 3.5 Synthesis
# --------------------------------------------------------------------------- #
def window_summary(y: pd.Series, windows: dict[str, pd.Series], low_threshold: float = 20.0) -> pd.DataFrame:
    """WHAT: for each named operating window (boolean mask), how many batches already run in it,
    their share of production, their mean stability and the share below `low_threshold`;
    plus an "all batches" row for reference.
    WHY: evidence for a recommended operating window that needs no model: what happened to
    the batches that were *already* produced under those conditions.
    """
    rows = {"all batches": pd.Series(True, index=y.index), **windows}
    out = {name: {"batches": int(m.sum()), "share_of_production": m.mean(),
                  "mean_stability": y[m].mean(), f"share_below_{low_threshold:g}": (y[m] < low_threshold).mean()}
           for name, m in rows.items()}
    return pd.DataFrame(out).T
