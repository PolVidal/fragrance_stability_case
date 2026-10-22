"""Data audit helpers (phase 1): describe the raw data without changing it.

Every function here only *reads* the DataFrame and returns a summary table.
Nothing is dropped, imputed or corrected: that is phase 2 (cleaning.py).
Each docstring states WHAT the function returns and WHY the check matters.
"""
from __future__ import annotations

import pandas as pd

from fragrance import config


# --------------------------------------------------------------------------- #
# Loading and schema
# --------------------------------------------------------------------------- #
def load_raw(path=config.RAW_PARQUET) -> pd.DataFrame:
    """Load the raw Parquet file exactly as delivered.

    WHAT: the untouched raw table (timestamps are still strings).
    WHY: the audit must see the data as it arrived. Any parsing or type
    conversion is itself a decision, so it happens explicitly in the
    notebook or in cleaning.py, where it can be logged.
    """
    return pd.read_parquet(path)


def schema_vs_statement(df: pd.DataFrame) -> pd.DataFrame:
    """Compare the columns in the file with the columns the code expects.

    WHAT: one row per column name with flags `in_file` and `in_config`.
    WHY: the statement and the delivered file can disagree (renamed or extra
    columns). Finding that first avoids silently analysing the wrong column.
    """
    expected = set(config.ID_COLS + config.CATEGORICAL_COLS + config.SENSOR_COLS
                   + [config.TEMP_UNIT, config.TARGET] + config.LEAKAGE_COLS)
    names = sorted(set(df.columns) | expected)
    return pd.DataFrame({
        "column": names,
        "in_file": [c in df.columns for c in names],
        "in_config": [c in expected for c in names],
        "dtype": [str(df[c].dtype) if c in df.columns else None for c in names],
    })


# --------------------------------------------------------------------------- #
# Q1. Grain: one row per batch, or several readings per batch?
# --------------------------------------------------------------------------- #
def rows_per_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Grain check.

    WHAT: one row per `batch_id` with the number of rows it has and how many
    distinct values it takes for the target, the reactor and each sensor.
    WHY: if batches have several rows, a row-level train/test split leaks
    the label (same batch in train and test).
    NOTE: "size" counts all rows; "nunique" ignores NaN, so a sensor column
    can show 0 distinct values for a batch whose only reading is missing.
    """
    return (
        df.groupby("batch_id")
          .agg(
              n_rows=("batch_id", "size"),                  # how many rows per batch
              n_target=(config.TARGET, "nunique"),          # distinct target values
              n_reactor=("production_line_id", "nunique"),  # distinct reactors
              n_temp=("temp_value", "nunique"),             # distinct sensor values
              n_pressure=("vessel_pressure_bar", "nunique"),
          )
    )


# --------------------------------------------------------------------------- #
# Q2-Q4. Structure, duplicates, missing values
# --------------------------------------------------------------------------- #
def timestamp_formats(ts: pd.Series) -> pd.DataFrame:
    """WHAT: one row per timestamp format found in `ts` (see
    config.TIMESTAMP_FORMATS), with count, share, whether it carries a time of
    day, and one example. Strings matching no known format are "unknown".
    WHY: values that fail a single-format parse are not necessarily lost: they
    may be the same date written another way. Knowing the formats tells us
    whether they can be recovered (phase 2) and how much precision is lost.
    """
    label = pd.Series("unknown", index=ts.index)
    for name, fmt in config.TIMESTAMP_FORMATS.items():
        ok = pd.to_datetime(ts, format=fmt, errors="coerce").notna() & (label == "unknown")
        label[ok] = name
    out = ts.groupby(label).agg(n="size", example="first")
    out["share"] = out["n"] / len(ts)
    out["has_time"] = out.index == "iso"
    return out.sort_values("n", ascending=False)


def parse_timestamps(ts: pd.Series) -> pd.Series:
    """WHAT: `ts` parsed to datetime, trying each known format in order;
    NaT where none matches. Date-only formats get midnight as time.
    WHY: needed for every time-based check (date range, drift, time split).
    Returns a new Series; the input DataFrame is not modified.
    """
    parsed = pd.Series(pd.NaT, index=ts.index, dtype="datetime64[ns]")
    for fmt in config.TIMESTAMP_FORMATS.values():
        parsed = parsed.fillna(pd.to_datetime(ts, format=fmt, errors="coerce"))
    return parsed


def physical_reactor(df: pd.DataFrame) -> pd.Series:
    """WHAT: a reactor key unique across plants, e.g. "Plant-01/R05".
    WHY: `production_line_id` values are reused in several plants, so
    "Reactor-05" alone mixes physically different machines.
    """
    return df["plant_location"] + "/R" + df["production_line_id"].str[-2:]


def category_mapping(df: pd.DataFrame, rows: str, cols: str) -> pd.DataFrame:
    """WHAT: row counts for every combination of two categorical columns
    (a contingency table).
    WHY: shows whether one column determines the other (one non-zero cell per
    row = nested or redundant), which changes how plant and reactor effects
    can be separated.
    """
    return pd.crosstab(df[rows], df[cols])


def batches_per_day(df: pd.DataFrame, ts: pd.Series, by: pd.Series) -> pd.DataFrame:
    """WHAT: number of batches per group (rows) and calendar day (columns).
    WHY: reveals downtime, late starts and uneven load per reactor, which
    matter for drift analysis and for a time-based train/test split.
    """
    return pd.crosstab(by, ts.dt.normalize())


def categorical_values(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """WHAT: every distinct value of each text column with its count
    (missing included) and a flag for hidden variants: values that differ
    only by case or surrounding spaces ("Plant-01" vs "plant-01 ").
    WHY: such variants silently split one category into several, which
    corrupts every group-by, and are a typical form of injected corruption.
    """
    parts = []
    for col in cols:
        vc = df[col].value_counts(dropna=False).rename_axis("value").reset_index(name="n")
        normalised = vc["value"].astype("string").str.strip().str.lower()
        vc["hidden_variant"] = normalised.duplicated(keep=False) & vc["value"].notna()
        vc.insert(0, "column", col)
        parts.append(vc)
    return pd.concat(parts, ignore_index=True)


def duplicate_report(df: pd.DataFrame) -> pd.Series:
    """WHAT: counts of (a) fully identical rows, (b) rows identical except
    for `batch_id` (the same record re-ingested under a new ID), and
    (c) repeated `batch_id`s.
    WHY: duplicates over-weight some batches and can put the same record in
    train and test. Case (b) is the one a plain `duplicated()` misses.
    """
    return pd.Series({
        "identical rows": int(df.duplicated().sum()),
        "identical except batch_id": int(df.drop(columns="batch_id").duplicated().sum()),
        "repeated batch_id": int(df["batch_id"].duplicated().sum()),
    })


def missing_report(df: pd.DataFrame, cols: list[str], by: pd.Series | None = None) -> pd.DataFrame:
    """WHAT: share of missing values per column; with `by`, one row per group
    (e.g. per plant or per day).
    WHY: missingness that concentrates in one reactor or period is not random
    (MAR/MNAR) and may itself carry information; uniform missingness (MCAR)
    can be imputed or tolerated without biasing the analysis.
    """
    missing = df[cols].isna()
    return missing.mean().to_frame("share_missing") if by is None else missing.groupby(by).mean()


def missing_per_row(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """WHAT: how many rows have 0, 1, 2, ... of `cols` missing at once.
    WHY: separates two failure mechanisms: one sensor dropping a value
    (scattered) versus the whole record being lost (all sensors at once).
    If sensors failed independently, rows with every sensor missing would
    be practically impossible (0.09 ** 7 ~ 5 in a million).
    """
    return df[cols].isna().sum(axis=1).value_counts().sort_index().rename("rows")


# --------------------------------------------------------------------------- #
# Q5. Values vs normal ranges: sensor errors or process anomalies?
# --------------------------------------------------------------------------- #
def temperature_celsius(df: pd.DataFrame) -> pd.Series:
    """WHAT: temperature in °C, converting the Fahrenheit readings
    (C = (F - 32) * 5/9). Sentinel codes are kept as they are.
    WHY: a correct 185 °F (85 °C) would otherwise look like a wildly
    out-of-range process anomaly. Returns a new Series; df is not modified.
    """
    t = df[config.TEMP_VALUE]
    is_f = df[config.TEMP_UNIT].eq("F") & ~t.isin(config.SENTINELS["temp_value"])
    return t.where(~is_f, (t - 32) * 5 / 9)


def sensor_status(df: pd.DataFrame, col: str) -> pd.Series:
    """WHAT: one label per row for sensor `col`: missing, sentinel,
    impossible (negative where physics forbids it), below, normal or above
    the normal range from the statement. Temperature is judged in °C.
    WHY: the label is the bridge between data quality and root cause:
    missing/sentinel/impossible are sensor errors, below/above are candidate
    process anomalies.
    """
    s = temperature_celsius(df) if col == config.TEMP_VALUE else df[col]
    lo, hi = config.NORMAL_RANGES[col]
    status = pd.Series("normal", index=df.index)
    status[s < lo] = "below"
    status[s > hi] = "above"
    if col in config.NON_NEGATIVE:
        status[s < 0] = "impossible"
    status[df[col].isin(config.SENTINELS.get(col, []))] = "sentinel"
    status[s.isna()] = "missing"
    return status


STATUS_ORDER = ["missing", "sentinel", "impossible", "below", "normal", "above"]


def range_report(df: pd.DataFrame) -> pd.DataFrame:
    """WHAT: for every sensor, the number of rows in each status of
    `sensor_status` (columns in a fixed order).
    WHY: one table that says, per sensor, how much is a sensor error and how
    much is a plausible out-of-range value worth investigating.
    """
    counts = {col: sensor_status(df, col).value_counts() for col in config.SENSOR_COLS}
    return pd.DataFrame(counts).T.reindex(columns=STATUS_ORDER).fillna(0).astype(int)


def target_by_status(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """WHAT: number of rows and mean target for each status of sensor `col`.
    WHY: the quickest test of "error or signal?". A sensor error should not
    move the target; a real process anomaly should.
    """
    out = df[config.TARGET].groupby(sensor_status(df, col)).agg(n="size", mean_stability="mean")
    return out.reindex([s for s in STATUS_ORDER if s in out.index])


def stuck_runs(df: pd.DataFrame, col: str, order: pd.Series, group: pd.Series,
               min_len: int = 3) -> pd.Series:
    """WHAT: number of runs (and rows in them) where `col` repeats exactly
    the same value on at least `min_len` consecutive batches of the same
    group, in `order` (time) order; plus the longest run.
    WHY: a frozen sensor keeps sending its last value. Those rows look
    valid but carry no information about the batch.
    """
    d = pd.DataFrame({"v": df[col], "g": group, "t": order}).sort_values(["g", "t"])
    new_run = d["v"].ne(d["v"].shift()) | d["g"].ne(d["g"].shift()) | d["v"].isna()
    run_len = d.groupby(new_run.cumsum())["v"].transform("size").where(d["v"].notna())
    return pd.Series({"runs": int(new_run.cumsum()[run_len >= min_len].nunique()),
                      "rows_in_runs": int((run_len >= min_len).sum()),
                      "longest_run": int(run_len.max())})


def spike_count(df: pd.DataFrame, col: str, order: pd.Series, group: pd.Series,
                window: int = 21, threshold: float = 5.0) -> int:
    """WHAT: number of readings far from their neighbours: robust z-score
    |x - rolling median| / (1.4826 * rolling MAD) above `threshold`,
    computed per group in time order. Sentinels and °F are excluded first.
    WHY: a spike is a single reading that jumps and comes straight back,
    typical of electrical noise. Median and MAD are used instead of mean and
    standard deviation because the spikes themselves would inflate those.
    """
    s = (temperature_celsius(df) if col == config.TEMP_VALUE else df[col])
    s = s.where(~df[col].isin(config.SENTINELS.get(col, [])))
    d = pd.DataFrame({"v": s, "g": group, "t": order}).sort_values(["g", "t"])
    rolling_median = lambda x: x.rolling(window, center=True, min_periods=5).median()
    med = d.groupby("g")["v"].transform(rolling_median)
    mad = (d["v"] - med).abs().groupby(d["g"]).transform(rolling_median)
    z = (d["v"] - med) / (1.4826 * mad)
    return int((z.abs() > threshold).sum())


# --------------------------------------------------------------------------- #
# Q6-Q7. Target and leakage
# --------------------------------------------------------------------------- #
def leakage_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    """WHAT: min, max and count of the target for each value of each derived
    column (`config.LEAKAGE_COLS`).
    WHY: if the target ranges of the values do not overlap, the column is a
    deterministic function of the target (a threshold on it) and would hand
    the model the answer: it must never be used as a feature.
    """
    parts = []
    for col in config.LEAKAGE_COLS:
        t = (df.groupby(col)[config.TARGET].agg(["min", "max", "size"])
               .sort_values("min").reset_index(names="value"))
        t.insert(0, "column", col)
        parts.append(t)
    return pd.concat(parts, ignore_index=True)
