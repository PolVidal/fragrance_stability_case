"""Plotting helpers shared by the notebooks. Figures are saved to reports/figures/."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from fragrance import config

# Single-hue sequential ramp (light = few, dark = many)
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def count_heatmap(table: pd.DataFrame, title: str, cbar_label: str, filename: str | None = None):
    """Heatmap of a table of groups x days (counts or means). Empty or zero cells are shown
    in white so that gaps such as downtime stand out."""
    fig, ax = plt.subplots(figsize=(14, 0.28 * len(table) + 1.5))
    data = table.where(table > 0)
    im = ax.imshow(data.values, aspect="auto", cmap=SEQUENTIAL_BLUE, interpolation="nearest")
    ax.set_yticks(range(len(table)), table.index, fontsize=8)
    step = 7
    ax.set_xticks(range(0, table.shape[1], step),
                  [pd.Timestamp(d).strftime("%b %d") for d in table.columns[::step]], fontsize=8)
    ax.set_title(title, loc="left", fontsize=11)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.025, pad=0.01)
    fig.tight_layout()
    if filename:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    return fig


# Categorical hues in fixed order (one per entity, never cycled); gray for context lines
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CONTEXT_GRAY = "#b4b2ab"
TEXT_SECONDARY = "#52514e"


def _style(ax, title: str, xlabel: str | None, ylabel: str | None):
    ax.set_title(title, loc="left", fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def _save(fig, filename: str | None, dpi: int = 150):
    fig.tight_layout()
    if filename:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.FIGURES_DIR / filename, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig


def histogram_with_thresholds(values: pd.Series, thresholds: dict[str, float], title: str,
                              xlabel: str, filename: str | None = None):
    """Histogram of one variable with labelled vertical reference lines (e.g. grade cut-offs)."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(values.dropna(), bins=60, color=CATEGORICAL[0], edgecolor="white", linewidth=0.5)
    top = ax.get_ylim()[1]
    for i, (label, x) in enumerate(thresholds.items()):  # stagger labels so close lines do not collide
        ax.axvline(x, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
        ax.text(x, top * (0.98 - 0.07 * i), f" {label}", fontsize=8, color=TEXT_SECONDARY, va="top")
    _style(ax, title, xlabel, "batches")
    return _save(fig, filename)


def lines_by_group(table: pd.DataFrame, title: str, ylabel: str, highlight: list[str] | None = None,
                   event: tuple[pd.Timestamp, str] | None = None, filename: str | None = None):
    """One line per column of `table` (index = time). With `highlight`, those columns get the
    categorical hues and the rest are drawn in gray as context. Lines are labelled at their end.
    `event` = (date, label) draws a vertical marker."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    coloured = highlight if highlight is not None else list(table.columns)
    for col in table.columns:
        if col in coloured:
            color, lw, z = CATEGORICAL[coloured.index(col)], 2, 3
        else:
            color, lw, z = CONTEXT_GRAY, 1.2, 2
        ax.plot(table.index, table[col], color=color, linewidth=lw, zorder=z,
                label=str(col) if highlight is not None and col in coloured else None)
        if col in coloured and highlight is None:  # end labels; highlighted lines use the legend
            ax.annotate(str(col), (table.index[-1], table[col].iloc[-1]), xytext=(4, 0),
                        textcoords="offset points", fontsize=8, color=TEXT_SECONDARY, va="center")
    if highlight is not None:
        ax.plot([], [], color=CONTEXT_GRAY, label="other reactors")
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    if event is not None:
        ax.axvline(event[0], color=TEXT_SECONDARY, linestyle="--", linewidth=1)
        ax.text(event[0], ax.get_ylim()[1], f" {event[1]}", fontsize=8, color=TEXT_SECONDARY, va="top")
    _style(ax, title, None, ylabel)
    return _save(fig, filename)




def control_charts(panels: dict[str, tuple[pd.DataFrame, pd.Series]], title: str,
                   baseline_end: pd.Timestamp | None = None, filename: str | None = None):
    """Shewhart control charts as small multiples, one panel per variable.
    Each panel gets (table, limits): `table` has one column per group (days x groups) and
    `limits` has centre / lcl / ucl. Points outside the limits are marked with a ring.
    The baseline period (used to compute the limits) is shaded."""
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 3.2 * len(panels)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (name, (table, lim)) in zip(axes, panels.items()):
        if baseline_end is not None:
            ax.axvspan(table.index.min(), baseline_end, color="#f0efec", zorder=0)
        for i, col in enumerate(table.columns):
            s = table[col]
            ax.plot(s.index, s, color=CATEGORICAL[i], linewidth=1.6, label=str(col), zorder=3)
            out = s[(s < lim["lcl"]) | (s > lim["ucl"])]
            ax.scatter(out.index, out, s=22, facecolors="none", edgecolors=CATEGORICAL[i],
                       linewidths=1, zorder=4)
        ax.axhline(lim["centre"], color=TEXT_SECONDARY, linewidth=1)
        for key in ["lcl", "ucl"]:
            ax.axhline(lim[key], color=TEXT_SECONDARY, linewidth=1, linestyle="--")
            ax.annotate(key.upper(), (1.0, lim[key]), xycoords=("axes fraction", "data"),
                        xytext=(3, 0), textcoords="offset points", fontsize=8,
                        color=TEXT_SECONDARY, va="center")
        _style(ax, name, None, None)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left", ncol=len(panels[next(iter(panels))][0].columns))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    return _save(fig, filename)


def curve_grid(curves: dict[str, pd.DataFrame], title: str, ylabel: str, ncols: int = 4,
               ranges: dict[str, tuple[float, float]] | None = None, filename: str | None = None):
    """Small multiples of binned curves (index = x, columns mean / ci_low / ci_high), one panel
    per variable, shared y axis so effect sizes are comparable across panels. Optional dashed
    lines mark each variable's normal range."""
    n = len(curves)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.9 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, c) in zip(axes, curves.items()):
        ax.fill_between(c.index, c["ci_low"], c["ci_high"], color=CATEGORICAL[0], alpha=0.2, linewidth=0)
        ax.plot(c.index, c["mean"], color=CATEGORICAL[0], linewidth=2, marker="o", markersize=4)
        if ranges and name in ranges:
            for x in ranges[name]:
                if c.index.min() <= x <= c.index.max():
                    ax.axvline(x, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
        _style(ax, name, None, ylabel if ax in axes[::ncols] else None)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    return _save(fig, filename)


def value_heatmap(table: pd.DataFrame, title: str, cbar_label: str, fmt: str = "{:.1f}",
                  filename: str | None = None):
    """Annotated heatmap of a small two-way table (e.g. mean stability by two binned sensors)."""
    fig, ax = plt.subplots(figsize=(1.2 * table.shape[1] + 2.5, 0.7 * table.shape[0] + 1.8))
    im = ax.imshow(table.values, cmap=SEQUENTIAL_BLUE, aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns, fontsize=8)
    ax.set_yticks(range(table.shape[0]), table.index, fontsize=8)
    ax.set_xlabel(table.columns.name, color=TEXT_SECONDARY)
    ax.set_ylabel(table.index.name, color=TEXT_SECONDARY)
    mid = np.nanmean(table.values)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.values[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                    color="white" if v > mid else "#0b0b0b")
    ax.set_title(title, loc="left", fontsize=11)
    fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.05, pad=0.02)
    return _save(fig, filename)


def importance_bars(values: pd.Series, title: str, xlabel: str, filename: str | None = None):
    """Horizontal bars, largest on top, value labels at the bar ends."""
    v = values.sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(v) + 1.2))
    ax.barh(v.index, v.values, color=CATEGORICAL[0], height=0.6)
    for i, x in enumerate(v.values):
        ax.text(x, i, f" {x:.2f}", va="center", fontsize=8, color=TEXT_SECONDARY)
    _style(ax, title, xlabel, None)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    return _save(fig, filename)


def dependence_grid(X: pd.DataFrame, shap_values: np.ndarray, features: list[str], title: str,
                    highlight: pd.Series | None = None, highlight_label: str = "",
                    filename: str | None = None):
    """SHAP dependence plots as small multiples: for each feature, its value (x) against its
    SHAP contribution in months (y). Optional boolean `highlight` colours a subset of points."""
    fig, axes = plt.subplots(1, len(features), figsize=(3.6 * len(features), 3.3), sharey=True)
    axes = np.atleast_1d(axes)
    cols = list(X.columns)
    for ax, f in zip(axes, features):
        x, s = X[f].to_numpy(), shap_values[:, cols.index(f)]
        base = np.ones(len(x), dtype=bool) if highlight is None else ~highlight.to_numpy()
        ax.scatter(x[base], s[base], s=4, color=CATEGORICAL[0], alpha=0.25, linewidths=0, label="other batches")
        if highlight is not None:
            h = highlight.to_numpy()
            ax.scatter(x[h], s[h], s=4, color=CATEGORICAL[1], alpha=0.5, linewidths=0, label=highlight_label)
        ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
        _style(ax, f, None, "SHAP contribution (months)" if ax is axes[0] else None)
    if highlight is not None:
        axes[0].legend(frameon=False, fontsize=8, markerscale=3, loc="lower right")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    return _save(fig, filename)


def metric_panels(table: pd.DataFrame, metrics: list[str], title: str, reference: list[str] | None = None,
                  filename: str | None = None):
    """One horizontal bar panel per metric (rows of `table` = models), sorted as given.
    Optional `reference` rows are drawn in gray (e.g. baselines)."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 0.5 * len(table) + 1.4), sharey=True)
    axes = np.atleast_1d(axes)
    labels = list(table.index)[::-1]
    for ax, m in zip(axes, metrics):
        vals = table[m].reindex(labels)
        colors = [CONTEXT_GRAY if lab in (reference or []) else CATEGORICAL[0] for lab in labels]
        ax.barh(labels, vals, color=colors, height=0.6)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.2f}", va="center", fontsize=8, color=TEXT_SECONDARY)
        _style(ax, m, "months", None)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    return _save(fig, filename)


def interval_examples(pred, low, high, truth, threshold: float, title: str, filename: str | None = None):
    """A sample of batches sorted by prediction: the interval as a vertical bar, the prediction
    as a dot and the true value as a ring; a dashed line marks the decision threshold."""
    order = np.argsort(pred)
    pred, low, high, truth = (np.asarray(a)[order] for a in (pred, low, high, truth))
    x = np.arange(len(pred))
    inside = (truth >= low) & (truth <= high)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.vlines(x, low, high, color=CATEGORICAL[0], alpha=0.35, linewidth=4, label="90% interval")
    ax.scatter(x, pred, s=14, color=CATEGORICAL[0], zorder=3, label="prediction")
    ax.scatter(x[inside], truth[inside], s=30, facecolors="none", edgecolors=TEXT_SECONDARY, zorder=4,
               label="true value (lab)")
    ax.scatter(x[~inside], truth[~inside], s=30, facecolors="none", edgecolors=CATEGORICAL[1], linewidths=1.5,
               zorder=4, label="true value outside interval")
    ax.axhline(threshold, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    ax.text(0, threshold, " Technical below this line", fontsize=8, color=TEXT_SECONDARY, va="bottom")
    ax.set_xticks([])
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=4)
    _style(ax, title, "test batches, sorted by prediction", "stability (months)")
    return _save(fig, filename)

POSITIVE, NEGATIVE = "#2a78d6", "#e34948"   # diverging pair: raises / lowers stability


def contribution_panels(panels: dict[str, pd.Series], title: str, filename: str | None = None):
    """Per-batch explanations as small multiples: for each batch, one horizontal bar per sensor
    with its contribution in months (blue raises stability, red lowers it), sorted by size.
    Panel titles carry the prediction and interval."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.4), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (name, s) in zip(axes, panels.items()):
        s = s.sort_values(key=np.abs)
        ax.barh(s.index, s.values, color=[POSITIVE if v > 0 else NEGATIVE for v in s.values], height=0.6)
        for i, v in enumerate(s.values):
            ax.text(v, i, f" {v:+.2f} " if v > 0 else f" {v:+.2f} ", va="center", ha="left" if v > 0 else "right",
                    fontsize=8, color=TEXT_SECONDARY)
        ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.8)
        _style(ax, name, "contribution (months)", None)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    lim = max(abs(a) for a in axes[0].get_xlim()) * 1.25
    axes[0].set_xlim(-lim, lim)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Compact figures with large text (saved at 200 dpi), readable when shown on a screen or a slide.
INK = "#0b0b0b"
TECHNICAL, AMBER, GREEN = "#e34948", "#eda100", "#1baf7a"


def _compact_axes(ax, title: str | None = None, grid: str | None = "y"):
    if title:
        ax.set_title(title, loc="left", fontsize=12, color=INK)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(axis=grid, color="#e6e5e0", linewidth=0.8)
        ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)


def approach_diagram(steps: list[tuple[str, str]], filename: str | None = None):
    """The project as a row of boxes joined by arrows: (title, description) per phase."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.set_xlim(0, 3 * len(steps)); ax.set_ylim(0, 2.6); ax.axis("off")
    for i, (title, body) in enumerate(steps):
        x = 0.2 + i * 3.0
        ax.add_patch(FancyBboxPatch((x, 0.35), 2.45, 1.9, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor="#eef4fc", edgecolor=CATEGORICAL[0], linewidth=1.5))
        ax.text(x + 0.18, 1.85, title, fontsize=14, fontweight="bold", color=INK, va="center")
        ax.text(x + 0.18, 1.05, body, fontsize=11, color=TEXT_SECONDARY, va="center", linespacing=1.4)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 2.95, 1.3), xytext=(x + 2.5, 1.3),
                        arrowprops=dict(arrowstyle="-|>", color=TEXT_SECONDARY, lw=1.5))
    return _save(fig, filename, dpi=200)


def time_split_diagram(periods: pd.DataFrame, event: pd.Timestamp, walk_forward: tuple[pd.Timestamp, pd.Timestamp],
                       filename: str | None = None):
    """Timeline of the time split. `periods` has one row per period with columns
    start, end (exclusive), label and color; `walk_forward` = (first, last) day of the folds."""
    import matplotlib.dates as mdates
    fig, ax = plt.subplots(figsize=(12, 2.4))
    for p in periods.itertuples():
        ax.barh(0, p.end - p.start, left=p.start, height=0.5, color=p.color)
        ax.text(p.start + (p.end - p.start) / 2, 0, p.label, ha="center", va="center", fontsize=10,
                color="white" if p.color != "#86b6ef" else INK)
    ax.axvline(event, color=INK, linestyle="--", linewidth=1.2, ymin=0.3, ymax=0.92)
    ax.text(event, 0.42, f" {event:%b} {event.day} event", fontsize=10, color=INK, va="bottom")
    ax.annotate("", xy=(walk_forward[1], -0.33), xytext=(walk_forward[0], -0.33),
                arrowprops=dict(arrowstyle="<->", color=TEXT_SECONDARY, lw=1))
    ax.text(walk_forward[0] + (walk_forward[1] - walk_forward[0]) / 2, -0.38, "walk-forward checks: 3 × 10 days",
            fontsize=9, color=TEXT_SECONDARY, ha="center", va="top")
    ax.set_ylim(-0.6, 0.65); ax.set_yticks([])
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b 1"))
    _compact_axes(ax, grid=None)
    ax.spines["left"].set_visible(False)
    return _save(fig, filename, dpi=200)




def error_vs_signal(error: pd.Series, signal: pd.Series, error_xlabel: str, signal_xlabel: str,
                    filename: str | None = None):
    """Two bar panels of mean stability: left, groups split by a sensor error (expected: no effect);
    right, groups split by a real value (expected: an effect). Bars in `signal` whose label
    contains '<' or '>' are drawn in red as out of range."""
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 3.0), sharey=True, gridspec_kw={"width_ratios": [2, 3]})
    panels = [(axes[0], error, [CONTEXT_GRAY] * len(error), "Sensor error: no effect", "{:.2f}", error_xlabel),
              (axes[1], signal, [TECHNICAL if ("<" in g or ">" in g) else CATEGORICAL[0] for g in signal.index],
               "Real value: real effect", "{:.1f}", signal_xlabel)]
    for ax, values, colors, title, fmt, xlabel in panels:
        bars = ax.bar(range(len(values)), values.values, color=colors, width=0.6)
        ax.set_xticks(range(len(values)), values.index, fontsize=9)
        for b, v in zip(bars, values.values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.08, fmt.format(v), ha="center", fontsize=11, color=INK)
        ax.set_ylim(18, 22.5)
        ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=9)
        _compact_axes(ax, title)
    axes[0].set_ylabel("Mean stability (months)", color=TEXT_SECONDARY)
    return _save(fig, filename, dpi=200)


def share_heatmap(table: pd.DataFrame, title: str, vmax: float = 25, filename: str | None = None):
    """Annotated heatmap of percentages (rows x columns), on a fixed 0-vmax scale so that a
    flat table looks flat."""
    fig, ax = plt.subplots(figsize=(4.8, 3.9))
    ax.imshow(table.values, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, f"{table.iat[i, j]:.1f}", ha="center", va="center", fontsize=10, color=INK)
    ax.set_xticks(range(table.shape[1]), table.columns, fontsize=10)
    ax.set_yticks(range(table.shape[0]), table.index, fontsize=10)
    ax.tick_params(length=0, colors=TEXT_SECONDARY)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    return _save(fig, filename, dpi=200)


def tree_diagram(tree, X: pd.DataFrame, y: pd.Series, labels: dict[str, tuple[str, str]],
                 low: float = 20.0, filename: str | None = None):
    """A fitted regression tree drawn as boxes. Split nodes show the question; leaves show the
    mean of `y`, the share below `low` and the number of rows of `X` that reach them, coloured
    red (> 50% below `low`), amber (10-50%) or green (< 10%). `labels` maps a feature to
    (short name, unit)."""
    t = tree.tree_
    leaf = pd.Series(tree.apply(X), index=X.index)
    stats = pd.DataFrame({"mean": y.groupby(leaf).mean(), "low": (y < low).groupby(leaf).mean(), "n": leaf.value_counts()})
    features = list(X.columns)
    xpos, leaves = {}, []

    def collect(node):
        if t.children_left[node] == -1:
            leaves.append(node)
        else:
            collect(t.children_left[node]); collect(t.children_right[node])

    def place(node):
        if t.children_left[node] == -1:
            xpos[node] = leaves.index(node)
        else:
            xpos[node] = (place(t.children_left[node]) + place(t.children_right[node])) / 2
        return xpos[node]

    collect(0); place(0)
    depth = tree.get_depth()
    fig, ax = plt.subplots(figsize=(12, 1.4 * (depth + 1)))
    ax.axis("off")

    def draw(node, level):
        x, yy = xpos[node], depth - level
        if t.children_left[node] == -1:
            s = stats.loc[node]
            color = TECHNICAL if s["low"] > 0.5 else (AMBER if s["low"] > 0.1 else GREEN)
            ax.text(x, yy, f"{s['mean']:.1f} months\n{s['low']:.0%} Technical\n{s['n'] / 1000:.0f}k batches",
                    ha="center", va="center", fontsize=10, color=INK,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.25, edgecolor=color))
            return
        name, unit = labels[features[t.feature[node]]]
        ax.text(x, yy, f"{name} ≤ {t.threshold[node]:.2f} {unit}?", ha="center", va="center", fontsize=11, color=INK,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#eef4fc", edgecolor=CATEGORICAL[0]))
        for child, answer in [(t.children_left[node], "yes"), (t.children_right[node], "no")]:
            ax.annotate("", xy=(xpos[child], yy - 1 + 0.28), xytext=(x, yy - 0.2),
                        arrowprops=dict(arrowstyle="-", color=TEXT_SECONDARY, lw=1))
            ax.text((x + xpos[child]) / 2, yy - 0.48, answer, fontsize=9, color=TEXT_SECONDARY, ha="center",
                    va="center", bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"))
            draw(child, level + 1)

    draw(0, 0)
    ax.set_xlim(-0.6, len(leaves) - 0.4)
    ax.set_ylim(-0.6, depth + 0.4)
    return _save(fig, filename, dpi=200)


def gap_comparison(raw: pd.Series, adjusted: pd.Series, title: str, filename: str | None = None):
    """Paired horizontal bars per group: raw gap (gray) and gap after adjustment (blue)."""
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    pos = np.arange(len(raw))
    ax.barh(pos - 0.2, raw.values, height=0.38, color=CONTEXT_GRAY, label="Raw gap")
    ax.barh(pos + 0.2, adjusted.values, height=0.38, color=CATEGORICAL[0], label="At equal operating conditions")
    for i, (r, a) in enumerate(zip(raw.values, adjusted.values)):
        ax.text(r + 0.02, i - 0.2, f"{r:+.2f}", va="center", ha="left", fontsize=10, color=INK)
        ax.text(0.03, i + 0.2, f"{a:+.2f}", va="center", ha="left", fontsize=10, color=CATEGORICAL[0])
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_yticks(pos, raw.index)
    ax.invert_yaxis()
    ax.set_xlim(min(raw.min(), 0) * 1.25, 0.25)
    ax.set_xlabel("Months vs Plant-01", color=TEXT_SECONDARY)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.45, -0.28), ncol=2)
    _compact_axes(ax, title, grid="x")
    return _save(fig, filename, dpi=200)


def stacked_gaps(parts: pd.DataFrame, total: pd.Series, title: str, xlabel: str, filename: str | None = None):
    """Each row's gap as stacked horizontal segments (columns of `parts`), with the total at the end."""
    colors = [CATEGORICAL[0], CATEGORICAL[1], CONTEXT_GRAY, CATEGORICAL[2]]
    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    left = np.zeros(len(parts))
    for col, color in zip(parts.columns, colors):
        ax.barh(parts.index, parts[col], left=left, color=color, label=col, height=0.55)
        left += parts[col].to_numpy()
    for i, v in enumerate(total.values):
        ax.text(v - 0.02, i, f"{v:+.2f}", va="center", ha="right", fontsize=10, color=INK)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.invert_yaxis()
    ax.set_xlim(total.min() * 1.3, 0.05)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.45, -0.28), ncol=len(parts.columns))
    _compact_axes(ax, title, grid="x")
    return _save(fig, filename, dpi=200)


def model_lines(table: pd.DataFrame, title: str, ylabel: str, highlight: str, filename: str | None = None):
    """One line per row of `table` (models) across its columns (periods), labelled at the end;
    the `highlight` row is drawn thicker."""
    palette = [CONTEXT_GRAY, "#86b6ef", CATEGORICAL[1], CATEGORICAL[2]]
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    cols = [str(c) for c in table.columns]
    others = [m for m in table.index if m != highlight]
    for name, color in list(zip(others, palette)) + [(highlight, CATEGORICAL[0])]:
        values = table.loc[name].to_numpy()
        ax.plot(cols, values, marker="o", color=color, linewidth=2.5 if name == highlight else 1.5)
        ax.text(len(cols) - 1 + 0.08, values[-1], name, va="center", fontsize=10, color=color)
    ax.set_xlim(-0.2, len(cols) - 1 + 0.9)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    _compact_axes(ax, title)
    return _save(fig, filename, dpi=200)


def tier_bars(share: pd.Series, technical: pd.Series, title: str, filename: str | None = None):
    """Red / amber / green bars: share of batches per tier, with the share truly Technical in the label."""
    colors = {"red": TECHNICAL, "amber": AMBER, "green": GREEN}
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    bars = ax.bar(range(len(share)), share.values * 100, color=[colors[t] for t in share.index], width=0.6)
    for b, v in zip(bars, share.values):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1.5, f"{v:.0%} of batches", ha="center", fontsize=10, color=INK)
    ax.set_xticks(range(len(share)), [f"{t.capitalize()}\n{technical[t]:.1%} Technical" for t in share.index], fontsize=10)
    ax.set_ylim(0, share.max() * 125)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    _compact_axes(ax, title, grid=None)
    return _save(fig, filename, dpi=200)


def threshold_lines(table: pd.DataFrame, title: str, xlabel: str, reference: float | None = None,
                    filename: str | None = None):
    """Recall and precision (columns of `table`, as fractions) against a threshold (index)."""
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    for col, color in [("recall", CATEGORICAL[0]), ("precision", CATEGORICAL[1])]:
        ax.plot(table.index, table[col] * 100, marker="o", color=color, linewidth=2)
        ax.text(table.index[-1] + 0.05, table[col].iloc[-1] * 100, col.capitalize(), va="center", fontsize=10, color=color)
    if reference is not None:
        ax.axvline(reference, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
        ax.text(reference + 0.03, 45, "grade cut-off", fontsize=9, color=TEXT_SECONDARY)
    ax.set_xticks(table.index)
    ax.set_xlim(table.index.min() - 0.15, table.index.max() + 0.6)
    ax.set_ylim(40, 100)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
    ax.set_ylabel("%", color=TEXT_SECONDARY)
    _compact_axes(ax, title)
    return _save(fig, filename, dpi=200)


def group_error_panels(panels: dict[str, pd.Series], overall: float, ylabel: str, filename: str | None = None):
    """Small multiples of bars (one panel per grouping) with a dashed line at the overall value."""
    widths = [max(len(s), 3) for s in panels.values()]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2, 2.8), sharey=True, gridspec_kw={"width_ratios": widths})
    for ax, (title, s) in zip(axes, panels.items()):
        bars = ax.bar(list(s.index), s.values, color=CATEGORICAL[0], width=0.6)
        for b, v in zip(bars, s.values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=INK)
        ax.axhline(overall, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
        ax.tick_params(axis="x", labelsize=9)
        _compact_axes(ax, title)
    axes[0].set_ylabel(ylabel, color=TEXT_SECONDARY)
    axes[0].set_ylim(0, max(s.max() for s in panels.values()) * 1.1)
    axes[-1].text(len(list(panels.values())[-1]) - 0.6, axes[0].get_ylim()[1] * 0.94,
                  f"dashed line: overall {overall:.2f}", fontsize=9, color=TEXT_SECONDARY, ha="right")
    return _save(fig, filename, dpi=200)


def grouped_bars(table: pd.DataFrame, title: str, ylabel: str, reference: float | None = None,
                 ylim: tuple[float, float] | None = None, filename: str | None = None):
    """Grouped vertical bars: one group per column of `table`, one bar per row (the last row in
    the accent colour), with an optional dashed reference line."""
    palette = [CONTEXT_GRAY, "#86b6ef", CATEGORICAL[0]][-len(table):]
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    x = np.arange(table.shape[1])
    w = 0.8 / len(table)
    for k, (name, row) in enumerate(table.iterrows()):
        ax.bar(x + (k - (len(table) - 1) / 2) * w, row.values, width=w, color=palette[k], label=name)
    if reference is not None:
        ax.axhline(reference, color=INK, linestyle="--", linewidth=1)
        ax.text(x[-1] + 0.45, reference + 0.5, f"target {reference:g}%", fontsize=9, color=INK, ha="right", va="bottom")
    ax.set_xticks(x, table.columns, fontsize=9)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(table))
    _compact_axes(ax, title)
    return _save(fig, filename, dpi=200)


def acute_vs_chronic(sources: pd.Series, levels: pd.DataFrame, filename: str | None = None):
    """Two panels. Left: Technical batches by source (e.g. an event vs normal operation), with the
    share of the total. Right: mean stability of groups of batches (rows of `levels`, columns
    `mean_stability` and `share_below_20`), the first row being the reference."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    colors = [TECHNICAL, CATEGORICAL[0]]
    bars = ax.barh(sources.index, sources.values / 1000, color=colors[:len(sources)], height=0.55)
    for b, v in zip(bars, sources.values):
        ax.text(b.get_width() + 0.8, b.get_y() + b.get_height() / 2, f"{v / 1000:.1f}k ({v / sources.sum():.0%})",
                va="center", fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, sources.max() / 1000 * 1.45)
    ax.set_xlabel("Technical-grade batches (thousands)", color=TEXT_SECONDARY)
    _compact_axes(ax, "Where Technical batches come from", grid="x")
    ax = axes[1]
    colors = [CONTEXT_GRAY] + [CATEGORICAL[0]] * (len(levels) - 1)
    bars = ax.bar(range(len(levels)), levels["mean_stability"], color=colors, width=0.6)
    for b, (m, low) in zip(bars, levels[["mean_stability", "share_below_20"]].itertuples(index=False)):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.1, f"{m:.1f} mo\n{low:.1%} Technical", ha="center",
                va="bottom", fontsize=9, color=INK)
    ax.set_xticks(range(len(levels)), levels.index, fontsize=9)
    ax.set_ylim(18, levels["mean_stability"].max() + 2)
    ax.set_ylabel("Mean stability (months)", color=TEXT_SECONDARY)
    _compact_axes(ax, "Normal operation vs batches already run well")
    return _save(fig, filename, dpi=200)
